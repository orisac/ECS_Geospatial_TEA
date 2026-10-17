"""
adapted_TEA_model: techno-economic model of an electrochemical nitrogen-recovery reactor.

Adapted from the electrolytic reactor TEA of Orella et al. (EconomicCase). The model
computes the cell voltage from ohmic, mass-transfer and Butler-Volmer kinetic losses,
sizes the reactor from the required current, and returns the levelized cost of
product ($/kg) along with a breakdown of its contributions.

All parameters accept NumPy arrays, so the model can be evaluated over a spatial grid
or a parameter sweep in a single call.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import fsolve


class EconomicCase:
    # Physical constants
    FARADAY_CONSTANT = 96485       # C mol^-1
    GAS_CONSTANT = 8.3140          # J mol^-1 K^-1

    # Unit conversions
    G_TO_KG = 1000
    SECONDS_TO_YEARS = 365.25 * 24 * 60 * 60
    CM_TO_M = 100
    SECONDS_TO_HOURS = 3600
    WATT_TO_KW = 1000

    # Production rate at which the reference areal capital cost applies
    BASE_RATE = 8.6                # mol s^-1

    # Order of the entries in self.breakdown
    BREAKDOWN_LABELS = ['Electricity', 'Chemicals', 'Overhead', 'Balance of Plant',
                        'Reactor Housing', 'Catalyst', 'Membrane']

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params : dict, optional
            Overrides for the default parameters below, keyed by attribute name and
            given in the units listed next to each default.
        """
        # ---- Default parameters ----
        # Prices
        self.feed_price = 0.0               # $ kg^-1
        self.salt_price = 0.0               # $ kg^-1
        self.solvent_price = 0.001          # $ kg^-1
        self.electricity_price = 0.0612     # $ kWh^-1
        self.catalyst_price = 32000         # $ kg^-1
        self.additional_factor = 0.14       # overhead fraction of total cost

        # Electrochemistry
        self.standard_potential = -1.233    # V
        self.transfer_coefficient = 0.5     # -
        self.number_electrons = 2           # mol e / mol product
        self.waste_electrons = 2            # mol e / mol waste
        self.product_fe = 0.90              # -
        self.her_fe = 0.0                   # -
        self.current_density = -15000.0     # A m^-2
        self.exchange_current_density = -0.1  # A m^-2

        # Streams and species
        self.molality = 1                   # mol R kg solvent^-1
        self.molarity = 1000                # mol R m^-3
        self.reactant_mw = 1.01             # g mol^-1
        self.salt_mw = 98                   # g mol^-1
        self.prod_mw = 2.02                 # g mol^-1
        self.waste_mw = 100                 # g mol^-1
        self.electrolyte_ratio = 1          # mol salt / mol e
        self.kp = 0.000                     # separation cost, $ kg^-1 mixture

        # Reactor design and transport
        self.diffusion_coeff = 5e-9         # m^2 s^-1
        self.bl_thickness = 1e-5            # m
        self.production_rate = 8.6          # mol s^-1
        self.conversion = 0.5               # mol consumed / mol fed
        self.cell_gap = 0.02                # cm
        self.conductivity = 0.1             # S cm^-1
        self.temperature = 298.15           # K

        # Capital costs. reference_area_price is the $/m^2 capex at BASE_RATE for
        # proven electrolyzer technology (NREL scale correlation); base_price and
        # membrane_price only set the housing:membrane split of that cost.
        self.base_price = 1.0626e4          # $ m^-2
        self.reference_area_price = 1.0626e4  # $ m^-2
        self.base_bop_price = 1             # balance-of-plant cost multiplier
        self.membrane_price = 1000          # $ m^-2
        self.scale_area = np.log(423 / 385) / np.log(50000 / 1500)

        self.catalyst_loading = 0.01        # mg metal cm^-2
        self.lifetime = 7                   # years
        self.operating_days = 300           # days per year
        self.recycle_electrolyte = 0.99
        self.recycle_feed = 0.99
        self.recycle_solvent = 0.99
        self.cost_of_capital = 0.1

        # ---- Apply user-supplied overrides ----
        if params is not None:
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    warnings.warn(f"Unknown parameter '{key}' — ignored. Check spelling.")

        # ---- Convert inputs to SI ----
        self.lifetime = self.lifetime * self.SECONDS_TO_YEARS * self.operating_days / 365.25
        self.reactant_mw = self.reactant_mw / self.G_TO_KG
        self.salt_mw = self.salt_mw / self.G_TO_KG
        self.prod_mw = self.prod_mw / self.G_TO_KG
        self.catalyst_loading = self.catalyst_loading / self.G_TO_KG ** 2 * self.CM_TO_M ** 2

        # Conductivity tracks molarity through this fixed ratio (see vary)
        self.k = self.molarity / self.conductivity

        self.run_model()

    def vary(self, attr, value):
        """Set one parameter (in input units) and recalculate the model."""
        if attr == 'lifetime':
            value = np.asarray(value) * self.SECONDS_TO_YEARS * self.operating_days / 365.25
        elif attr in ('reactant_mw', 'prod_mw', 'salt_mw'):
            value = np.asarray(value) / self.G_TO_KG
        elif attr == 'catalyst_loading':
            value = np.asarray(value) / self.G_TO_KG ** 2 * self.CM_TO_M ** 2

        setattr(self, attr, value)

        if attr == 'molarity':
            self.conductivity = self.molarity / self.k

        self.run_model()

    # ------------------------------------------------------------------ #
    # Model steps
    # ------------------------------------------------------------------ #

    def run_model(self):
        """Evaluate every model step in order."""
        self._calculate_current()
        self._calculate_reactor_area()
        self._calculate_ohmic_losses()
        self._calculate_limiting_current_density()

        if np.any(np.abs(self.current_density) >= np.abs(self.limiting_current_density)):
            warnings.warn('Operating at or above limiting current density — check values.')

        self._calculate_mass_transfer_losses()
        self._calculate_kinetic_losses()
        self._calculate_open_circuit_voltage()
        self._calculate_cell_voltage()
        self._calculate_inlet_flow_rate()
        self._calculate_feed_flow_rate()
        self._calculate_capital_costs()
        self._calculate_bop_costs()
        self._calculate_separation_costs()
        self._calculate_electricity_costs()
        self._calculate_electrolyte_costs()
        self._calculate_total_cost()
        self._calculate_specific_energy()

    def _butler_volmer(self, overpotential):
        """Butler-Volmer rate expression, j / j0, as a function of overpotential [V]."""
        n_e = self.number_electrons
        F = self.FARADAY_CONSTANT
        alpha = self.transfer_coefficient
        RT = self.GAS_CONSTANT * self.temperature
        return (np.exp(-n_e * F * alpha * overpotential / RT)
                - np.exp(n_e * F * (1 - alpha) * overpotential / RT))

    def _calculate_current(self):
        """Total cell current [A] needed for the production rate at the product FE."""
        self.current = (-self.production_rate * self.number_electrons
                        * self.FARADAY_CONSTANT / self.product_fe)

    def _calculate_reactor_area(self):
        """Electrode area [m^2]."""
        self.reactor_area = self.current / self.current_density

    def _calculate_ohmic_losses(self):
        """Ohmic drop across the electrolyte gap [V]."""
        self.ohmics = (self.current_density * self.cell_gap
                       / self.conductivity / self.CM_TO_M ** 2)

    def _calculate_limiting_current_density(self):
        """Mass-transfer-limited current density [A m^-2] at the outlet concentration."""
        self.limiting_current_density = (
            -self.number_electrons * self.FARADAY_CONSTANT * self.diffusion_coeff
            * self.molarity * (1 - self.conversion) / self.bl_thickness
        )

    def _calculate_mass_transfer_losses(self):
        """Concentration overpotential [V]."""
        self.mass_transfer = (
            self.GAS_CONSTANT * self.temperature / self.number_electrons / self.FARADAY_CONSTANT
            * np.log(1 - self.current_density / self.limiting_current_density)
        )

    def _calculate_kinetic_losses(self):
        """Kinetic overpotential [V], solving j = j0 * BV(eta) for eta."""
        j = np.atleast_1d(self.current_density).copy()
        n0 = -0.1 * np.ones_like(j)
        sol = fsolve(lambda n: j - self.exchange_current_density * self._butler_volmer(n), n0)
        self.kinetics = sol if np.ndim(self.current_density) > 0 else float(sol[0])

    def _calculate_open_circuit_voltage(self):
        """Nernstian open-circuit voltage [V]."""
        self.ocv = (
            self.standard_potential
            + self.GAS_CONSTANT * self.temperature / self.number_electrons / self.FARADAY_CONSTANT
            * np.log((1 - self.conversion) / self.conversion / self.product_fe)
        )

    def _calculate_cell_voltage(self):
        """Cell voltage [V] as the sum of the open-circuit voltage and all losses."""
        self.cell_voltage = self.ocv + self.ohmics + self.mass_transfer + self.kinetics

    def _calculate_inlet_flow_rate(self):
        """Molar feed flow [mol/s] and volumetric flow [L/s] into the reactor."""
        waste_fe = 1 - self.product_fe - self.her_fe
        self.flow_rate = (
            -self.current / self.conversion / self.FARADAY_CONSTANT
            * (self.product_fe / self.number_electrons + waste_fe / self.waste_electrons)
        )
        self.volumetric_flow_rate = self.flow_rate / self.molarity * 1000

    def _calculate_feed_flow_rate(self):
        """Fresh reactant feed [mol/s] after recycling unconverted reactant."""
        self.feed_rate = self.flow_rate * (1 - self.recycle_feed * (1 - self.conversion))

    def _calculate_capital_costs(self):
        """Reactor housing, membrane and catalyst capital [$].

        The areal cost scales with production rate from reference_area_price at
        BASE_RATE, then splits between housing and membrane by base_price:membrane_price.
        """
        ln_p = np.log(self.reference_area_price) + self.scale_area * np.log(
            self.production_rate / self.BASE_RATE)
        total_cost = np.exp(ln_p) * self.reactor_area
        split = self.base_price + self.membrane_price
        self.capital_costs = total_cost * self.base_price / split
        self.membrane_costs = total_cost * self.membrane_price / split
        self.catalyst_costs = self.reactor_area * self.catalyst_loading * self.catalyst_price

    def _calculate_bop_costs(self):
        """Balance-of-plant capital [$]: pump plus power electronics.

        Pump cost uses the NREL correlation in gallons per minute (adjusted to 2026 $);
        power electronics cost $199 per kW of cell power.
        """
        gpm = np.clip(self.volumetric_flow_rate * 15.8503, 10, 900)
        pump_costs = 276.8688 / 193.533 * np.exp(8.28 - 0.29 * np.log(gpm) + 0.074 * np.log(gpm) ** 2)
        electronics_costs = 199 * (self.current * self.cell_voltage
                                   / self.WATT_TO_KW / self.SECONDS_TO_HOURS)
        self.bop_costs = (pump_costs + electronics_costs) * self.base_bop_price

    def _calculate_separation_costs(self):
        """Downstream separation cost [$/s] at kp $ per kg of outlet mixture."""
        prod_flow = self.production_rate * self.prod_mw
        waste_flow = (-self.current / self.FARADAY_CONSTANT
                      * (1 - self.product_fe - self.her_fe) / self.waste_electrons * self.waste_mw)
        hydrogen_flow = -self.current / self.FARADAY_CONSTANT * self.her_fe / 2 * 0.00202
        reactant_flow = self.flow_rate * (1 - self.conversion) * self.reactant_mw
        self.sep_costs = self.kp * (prod_flow + waste_flow + hydrogen_flow + reactant_flow)

    def _calculate_electricity_costs(self):
        """Electricity cost [$/s] for the cell plus pumping.

        Pumping power [W] = 9.8 * Q * sqrt(reactor area) / (0.4 * 0.95), where
        Q = 3 * flow_rate / molarity * 1e-3.
        """
        cell = (self.current * self.cell_voltage * self.electricity_price
                / self.WATT_TO_KW / self.SECONDS_TO_HOURS)
        pumping = (9.8 * (self.flow_rate * 3) / self.molarity * 1e-3 * np.sqrt(self.reactor_area)
                   / (0.4 * 0.95) * self.electricity_price / self.WATT_TO_KW / self.SECONDS_TO_HOURS)
        self.electricity_costs = cell + pumping

    def _calculate_electrolyte_costs(self):
        """Chemical make-up cost [$/s]: reactant feed, solvent (acid) and supporting salt."""
        reactant = self.feed_rate * self.reactant_mw * self.feed_price
        solvent = self.flow_rate * self.solvent_price * (1 - self.recycle_solvent) / self.molality
        salt = (-self.current / self.FARADAY_CONSTANT * self.electrolyte_ratio * self.salt_mw
                * self.salt_price * (1 - self.recycle_electrolyte))
        self.electrolyte_costs = reactant + solvent + salt

    def _calculate_total_cost(self):
        """Levelized product cost [$/kg] and its breakdown.

        Capital is annualized with a capital-recovery (annuity) factor over the plant
        lifetime and spread over operating seconds; overhead is applied as a fraction
        of the total cost.
        """
        op_seconds = self.SECONDS_TO_YEARS * self.operating_days / 365.25
        years = self.lifetime / op_seconds

        if self.cost_of_capital != 0:
            annuity_factor = self.cost_of_capital / (1 - (1 + self.cost_of_capital) ** (-years))

            def amortize(capex):
                return capex * annuity_factor / op_seconds
        else:
            def amortize(capex):
                return capex / self.lifetime

        per_kg = self.production_rate * self.prod_mw  # kg product / s

        opex = self.electricity_costs + self.electrolyte_costs + self.sep_costs
        capex = self.capital_costs + self.bop_costs + self.catalyst_costs + self.membrane_costs
        self.cost = (opex + amortize(capex)) / (1 - self.additional_factor) / per_kg

        electricity = self.electricity_costs / per_kg
        seps = self.sep_costs / per_kg
        electrolyte = self.electrolyte_costs / per_kg
        cell = amortize(self.capital_costs) / per_kg
        bop = amortize(self.bop_costs) / per_kg
        catalyst = amortize(self.catalyst_costs) / per_kg
        membrane = amortize(self.membrane_costs) / per_kg
        additional = self.cost - electricity - electrolyte - cell - bop - seps - catalyst - membrane

        # Rows follow BREAKDOWN_LABELS; broadcast so array inputs give a 2-D breakdown
        self.breakdown = np.array(np.broadcast_arrays(
            electricity, electrolyte, additional, bop, cell, catalyst, membrane))

    def _calculate_specific_energy(self):
        """Cell energy per mass of product [kJ/kg]."""
        self.specific_energy = (self.current * self.cell_voltage
                                / (self.production_rate * self.prod_mw) * 1e-3)


def load_product_cases(xlsx_path):
    """Build one EconomicCase per fertilizer product from TEA_inputs.xlsx.

    Returns
    -------
    products : pandas.DataFrame
        The 'products' sheet (one row per fertilizer).
    cases : list of EconomicCase
        Baseline parameters with each product's overrides applied, in sheet order.
    """
    base = pd.read_excel(xlsx_path, sheet_name='base_parameters')
    params = dict(zip(base['parameter'], base['value']))
    products = pd.read_excel(xlsx_path, sheet_name='products')

    cases = []
    for _, row in products.iterrows():
        case = EconomicCase(params)
        for attr in ('solvent_price', 'molality', 'prod_mw', 'production_rate'):
            case.vary(attr, row[attr])
        cases.append(case)
    return products, cases
