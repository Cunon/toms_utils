import sys
import numpy as np
import pandas as pd

import openmdao.api as om

import pycycle.api as pyc

import pandas as pd
import re

class MixedFlowTurbofan(pyc.Cycle):

    def setup(self):
        design = self.options['design']

        USE_TABULAR = False

        if USE_TABULAR: 
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = "FAR"
        else: 
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = "Jet-A(g)"

        self.add_subsystem('fc', pyc.FlightConditions())
        # Inlet Components
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('inlet_duct', pyc.Duct())
        # Fan Components - Split here for CFD integration Add a CFDStart Compomponent
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.AXI5,
                                             map_extrap=True),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('splitter', pyc.Splitter())
        # Core Stream components
        self.add_subsystem('splitter_core_duct', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap, map_extrap=True),
                                             promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('lpc_duct', pyc.Duct())
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap, 
                                        bleed_names=['cool1'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['cool3']))

        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
       
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap,
                                          bleed_names=['cool3'],map_extrap=True),promotes_inputs=[('Nmech','HP_Nmech')])
        self.add_subsystem('hpt_duct', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap,
                                        bleed_names=['cool1'],map_extrap=True), promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('lpt_duct', pyc.Duct())
        # Bypass Components
        self.add_subsystem('bypass_duct', pyc.Duct())
        # Mixer component
        self.add_subsystem('mixer', pyc.Mixer(designed_stream=1))
        self.add_subsystem('mixer_duct', pyc.Duct())
        # Afterburner Components
        self.add_subsystem('afterburner', pyc.Combustor(fuel_type=FUEL_TYPE))
    
        # End CFD HERE
        # Nozzle
        self.add_subsystem('mixed_nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cfg'))

        # Mechanical components
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','LP_Nmech')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','HP_Nmech')])

        # Aggregating component
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=2))

        # Connnect flow path
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'inlet_duct.Fl_I')
        self.pyc_connect_flow('inlet_duct.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        # Core connections
        self.pyc_connect_flow('splitter.Fl_O1', 'splitter_core_duct.Fl_I')
        self.pyc_connect_flow('splitter_core_duct.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'lpc_duct.Fl_I')
        self.pyc_connect_flow('lpc_duct.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'hpt_duct.Fl_I')
        self.pyc_connect_flow('hpt_duct.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'lpt_duct.Fl_I')
        self.pyc_connect_flow('lpt_duct.Fl_O','mixer.Fl_I1')
        # Bypass Connections
        self.pyc_connect_flow('splitter.Fl_O2', 'bypass_duct.Fl_I')
        self.pyc_connect_flow('bypass_duct.Fl_O', 'mixer.Fl_I2')

        #Mixer Connections
        self.pyc_connect_flow('mixer.Fl_O', 'mixer_duct.Fl_I')
        # After Burner
        self.pyc_connect_flow('mixer_duct.Fl_O','afterburner.Fl_I')

        # Nozzle
        self.pyc_connect_flow('afterburner.Fl_O','mixed_nozz.Fl_I')

        # Connect cooling flows
        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)

        # Make additional model connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('afterburner.Wfuel', 'perf.Wfuel_1')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('mixed_nozz.Fg', 'perf.Fg_0')

        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        self.connect('fc.Fl_O:stat:P', 'mixed_nozz.Ps_exhaust')

        # Add balence components to close the implicit components
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='lbf')
            self.connect('balance.W', 'fc.W')
            self.connect('perf.Fn', 'balance.lhs:W')
            # self.add_subsystem('wDV',IndepVarComp('wDes',100,units='lbm/s'))
            # self.connect('wDV.wDes','fc.W')

            balance.add_balance('BPR', eq_units=None, lower=0.25, val=5.0)
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.ER', 'balance.lhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8, eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:hpt_PR')
        else:

            balance.add_balance('W', lower=1e-3, upper=200., units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'fc.W')
            self.connect('mixed_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=0.25, upper=5.0, eq_units='psi')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('mixer.Fl_I1_calc:stat:P', 'balance.lhs:BPR')
            self.connect('bypass_duct.Fl_O:stat:P', 'balance.rhs:BPR')

            balance.add_balance('FAR_core', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_core', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR_core')

            balance.add_balance('FAR_ab', eq_units='degR', lower=1e-4, upper=.06, val=.017)
            self.connect('balance.FAR_ab', 'afterburner.Fl_I:FAR')
            self.connect('afterburner.Fl_O:tot:T', 'balance.lhs:FAR_ab')

            balance.add_balance('LP_Nmech', val=1., units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.LP_Nmech', 'LP_Nmech')
            self.connect('lp_shaft.pwr_in', 'balance.lhs:LP_Nmech')
            self.connect('lp_shaft.pwr_out', 'balance.rhs:LP_Nmech')

            balance.add_balance('HP_Nmech', val=1., units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.HP_Nmech', 'HP_Nmech')
            self.connect('hp_shaft.pwr_in', 'balance.lhs:HP_Nmech')
            self.connect('hp_shaft.pwr_out', 'balance.rhs:HP_Nmech')

        # Off design
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-10
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1


        self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()


def print_perf(prob,ptName):
    print('BPR',prob[ptName+'.balance.BPR'])
    print('W',prob[ptName+'.balance.W'])
    print('Fnet uninst.',prob[ptName+'.perf.Fn'])

def page_viewer(point):
    flow_stations = ['fc.Fl_O', 'inlet.Fl_O', 'inlet_duct.Fl_O', 'fan.Fl_O', 'bypass_duct.Fl_O',
                     'splitter.Fl_O2', 'splitter.Fl_O1', 'splitter_core_duct.Fl_O',
                     'lpc.Fl_O', 'lpc_duct.Fl_O', 'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                     'hpt.Fl_O', 'hpt_duct.Fl_O', 'lpt_duct.Fl_O',
                     'mixer.Fl_O', 'mixer_duct.Fl_O', 'afterburner.Fl_O', 'mixed_nozz.Fl_O']

    compressors = ['fan', 'hpc', 'lpc']
    burners = ['burner', 'afterburner']
    turbines = ['hpt', 'lpt']
    shafts = ['hp_shaft', 'lp_shaft']

    print('*'*80)
    print('* ' + ' '*10 + point)
    print('*'*80)
    print_perf(prob, point)

    pyc.print_flow_station(prob,[point+ "."+fl for fl in flow_stations])
    pyc.print_compressor(prob,[point+ "." + c for c in compressors])
    # print_splitter(prob,[point+ ".splitter" ])
    pyc.print_burner(prob,[point+ "." + b for b in burners])
    pyc.print_turbine(prob,[point+ "." + turb for turb in turbines])
    pyc.print_mixer(prob, [point+'.mixer'])
    pyc.print_nozzle(prob, [point + '.mixed_nozz'])
    pyc.print_shaft(prob, [point+ "." + s for s in shafts])
    pyc.print_bleed(prob, [point+'.hpc', point+'.bld3'])


class MPMixedFlowTurbofan(pyc.MPCycle):

    def setup(self):

        self.pyc_add_pnt('DESIGN', MixedFlowTurbofan(design=True, thermo_method='CEA'))

        self.set_input_defaults('DESIGN.balance.rhs:BPR', 1.05 ,units=None) # defined as 1 over 2
        self.set_input_defaults('DESIGN.inlet.MN', 0.751)
        self.set_input_defaults('DESIGN.inlet_duct.MN', 0.4463)
        self.set_input_defaults('DESIGN.fan.MN', 0.4578)
        self.set_input_defaults('DESIGN.splitter.MN1', 0.3104)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.4518)
        self.set_input_defaults('DESIGN.splitter_core_duct.MN', 0.3121)
        self.set_input_defaults('DESIGN.lpc.MN', 0.3059)
        self.set_input_defaults('DESIGN.lpc_duct.MN', 0.3563)
        self.set_input_defaults('DESIGN.hpc.MN', 0.2442)
        self.set_input_defaults('DESIGN.bld3.MN', 0.3000)
        self.set_input_defaults('DESIGN.burner.MN', 0.1025)
        self.set_input_defaults('DESIGN.hpt.MN', 0.3650)
        self.set_input_defaults('DESIGN.hpt_duct.MN', 0.3063)
        self.set_input_defaults('DESIGN.lpt.MN', 0.4127)
        self.set_input_defaults('DESIGN.lpt_duct.MN', 0.4463)
        self.set_input_defaults('DESIGN.bypass_duct.MN', 0.4463)
        self.set_input_defaults('DESIGN.mixer_duct.MN', 0.4463)
        self.set_input_defaults('DESIGN.afterburner.MN', 0.1025)
        self.set_input_defaults('DESIGN.LP_Nmech', 4666.1, units='rpm')
        self.set_input_defaults('DESIGN.HP_Nmech', 14705.7, units='rpm')

        # Refactored: Replaced global pyc_add_cycle_param with a shared parameters list.
        # This prevents OpenMDAO from stripping the component namespace prefix and hiding 
        # parameters from the pyc.print_burner() and pyc.print_shaft() viewers.
        shared_params = [
            ('balance.rhs:FAR_ab', 3400, 'degR'),
            ('hp_shaft.HPX', 250, 'hp'),
            ('inlet.ram_recovery', 0.9990, None),
            ('inlet_duct.dPqP', 0.0107, None),
            ('splitter_core_duct.dPqP', 0.0048, None),
            ('lpc_duct.dPqP', 0.0101, None),
            ('burner.dPqP', 0.0540, None),
            ('hpt_duct.dPqP', 0.0051, None),
            ('lpt_duct.dPqP', 0.0107, None),
            ('bypass_duct.dPqP', 0.0107, None),
            ('mixer_duct.dPqP', 0.0107, None),
            ('afterburner.dPqP', 0.0540, None),
            ('mixed_nozz.Cfg', 0.9933, None),
            ('hpc.cool1:frac_W', 0.050708, None),
            ('hpc.cool1:frac_P', 0.5, None),
            ('hpc.cool1:frac_work', 0.5, None),
            ('bld3.cool3:frac_W', 0.067214, None),
            ('hpt.cool3:frac_P', 1.0, None),
            ('lpt.cool1:frac_P', 1.0, None)
        ]

        # Apply shared parameters explicitly to the DESIGN point
        for param, val, units in shared_params:
            if units:
                self.set_input_defaults(f'DESIGN.{param}', val, units=units)
            else:
                self.set_input_defaults(f'DESIGN.{param}', val)

        self.od_pts = ['OD_35K_080M_T4_3100', 'OD_35K_080M_T4_3000', 'OD_35K_080M_T4_2900', 'OD_35K_080M_T4_2800', 'OD_35K_080M_T4_2700']
        self.od_T4s = [3100, 3000, 2900, 2800, 2700]
        self.od_alts = [35000, 35000, 35000, 35000, 35000]
        self.od_MNs = [0.8, 0.8, 0.8, 0.8, 0.8]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, MixedFlowTurbofan(design=False, thermo_method='CEA'))

            self.set_input_defaults(f'{pt}.balance.rhs:FAR_core', self.od_T4s[i], units='degR')
            self.set_input_defaults(f'{pt}.fc.alt', self.od_alts[i], units='ft')
            self.set_input_defaults(f'{pt}.fc.MN', self.od_MNs[i])

            # Apply shared parameters explicitly to the OD points
            for param, val, units in shared_params:
                if units:
                    self.set_input_defaults(f'{pt}.{param}', val, units=units)
                else:
                    self.set_input_defaults(f'{pt}.{param}', val)

        # map scalars
        self.pyc_connect_des_od('fan.s_PR', 'fan.s_PR')
        self.pyc_connect_des_od('fan.s_Wc', 'fan.s_Wc')
        self.pyc_connect_des_od('fan.s_eff', 'fan.s_eff')
        self.pyc_connect_des_od('fan.s_Nc', 'fan.s_Nc')
        self.pyc_connect_des_od('lpc.s_PR', 'lpc.s_PR')
        self.pyc_connect_des_od('lpc.s_Wc', 'lpc.s_Wc')
        self.pyc_connect_des_od('lpc.s_eff', 'lpc.s_eff')
        self.pyc_connect_des_od('lpc.s_Nc', 'lpc.s_Nc')
        self.pyc_connect_des_od('hpc.s_PR', 'hpc.s_PR')
        self.pyc_connect_des_od('hpc.s_Wc', 'hpc.s_Wc')
        self.pyc_connect_des_od('hpc.s_eff', 'hpc.s_eff')
        self.pyc_connect_des_od('hpc.s_Nc', 'hpc.s_Nc')
        self.pyc_connect_des_od('hpt.s_PR', 'hpt.s_PR')
        self.pyc_connect_des_od('hpt.s_Wp', 'hpt.s_Wp')
        self.pyc_connect_des_od('hpt.s_eff', 'hpt.s_eff')
        self.pyc_connect_des_od('hpt.s_Np', 'hpt.s_Np')
        self.pyc_connect_des_od('lpt.s_PR', 'lpt.s_PR')
        self.pyc_connect_des_od('lpt.s_Wp', 'lpt.s_Wp')
        self.pyc_connect_des_od('lpt.s_eff', 'lpt.s_eff')
        self.pyc_connect_des_od('lpt.s_Np', 'lpt.s_Np')

        # flow areas
        self.pyc_connect_des_od('mixed_nozz.Throat:stat:area', 'balance.rhs:W')
        self.pyc_connect_des_od('inlet.Fl_O:stat:area', 'inlet.area')
        self.pyc_connect_des_od('fan.Fl_O:stat:area', 'fan.area')
        self.pyc_connect_des_od('splitter.Fl_O1:stat:area', 'splitter.area1')
        self.pyc_connect_des_od('splitter.Fl_O2:stat:area', 'splitter.area2')
        self.pyc_connect_des_od('splitter_core_duct.Fl_O:stat:area', 'splitter_core_duct.area')
        self.pyc_connect_des_od('lpc.Fl_O:stat:area', 'lpc.area')
        self.pyc_connect_des_od('lpc_duct.Fl_O:stat:area', 'lpc_duct.area')
        self.pyc_connect_des_od('hpc.Fl_O:stat:area', 'hpc.area')
        self.pyc_connect_des_od('bld3.Fl_O:stat:area', 'bld3.area')
        self.pyc_connect_des_od('burner.Fl_O:stat:area', 'burner.area')
        self.pyc_connect_des_od('hpt.Fl_O:stat:area', 'hpt.area')
        self.pyc_connect_des_od('hpt_duct.Fl_O:stat:area', 'hpt_duct.area')
        self.pyc_connect_des_od('lpt.Fl_O:stat:area', 'lpt.area')
        self.pyc_connect_des_od('lpt_duct.Fl_O:stat:area', 'lpt_duct.area')
        self.pyc_connect_des_od('bypass_duct.Fl_O:stat:area', 'bypass_duct.area')
        self.pyc_connect_des_od('mixer.Fl_O:stat:area', 'mixer.area')
        self.pyc_connect_des_od('mixer.Fl_I1_calc:stat:area', 'mixer.Fl_I1_stat_calc.area')
        self.pyc_connect_des_od('mixer_duct.Fl_O:stat:area', 'mixer_duct.area')
        self.pyc_connect_des_od('afterburner.Fl_O:stat:area', 'afterburner.area')


        super().setup()


if __name__ == "__main__":
    import time
    from openmdao.api import Problem

    prob = Problem()

    prob.model = mp_mixedflow = MPMixedFlowTurbofan()

    prob.setup()

    #Define the design point
    prob.set_val('DESIGN.fan.PR', 3.3)
    prob.set_val('DESIGN.fan.eff', 0.8948)
    prob.set_val('DESIGN.lpc.PR', 1.935)
    prob.set_val('DESIGN.lpc.eff', 0.9243)
    prob.set_val('DESIGN.hpc.PR', 4.9)
    prob.set_val('DESIGN.hpc.eff', 0.8707)
    prob.set_val('DESIGN.hpt.eff', 0.8888)
    prob.set_val('DESIGN.lpt.eff', 0.8996)
    prob.set_val('DESIGN.fc.alt', 35000., units='ft') 
    prob.set_val('DESIGN.fc.MN', 0.8) 
    prob.set_val('DESIGN.balance.rhs:W', 5500.0, units='lbf')
    prob.set_val('DESIGN.balance.rhs:FAR_core', 3200, units='degR')

    # Set initial guesses for the balances
    prob['DESIGN.balance.FAR_core'] = 0.025
    prob['DESIGN.balance.FAR_ab'] = 0.025
    prob['DESIGN.balance.BPR'] = 1.0
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 3.5
    prob['DESIGN.balance.hpt_PR'] = 2.5
    prob['DESIGN.fc.balance.Pt'] = 5.2
    prob['DESIGN.fc.balance.Tt'] = 440.0
    prob['DESIGN.mixer.balance.P_tot']= 15

    for i,pt in enumerate(mp_mixedflow.od_pts):

        prob[pt+'.balance.FAR_core'] = 0.025
        prob[pt+'.balance.FAR_ab'] = 0.025
        prob[pt+'.balance.BPR'] = 2.5
        prob[pt+'.balance.W'] = 50.
        prob[pt+'.balance.HP_Nmech'] = 14000
        prob[pt+'.balance.LP_Nmech'] = 4000
        prob[pt+'.fc.balance.Pt'] = 5.2
        prob[pt+'.fc.balance.Tt'] = 440.0
        prob[pt+'.mixer.balance.P_tot']= 15
        prob[pt+'.hpt.PR'] = 2.0
        prob[pt+'.lpt.PR'] = 4.0
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()
    page_viewer('DESIGN')

   
    for pt in ['DESIGN']+mp_mixedflow.od_pts:
        page_viewer(pt)

    print()
    print("time", time.time() - st)

    # =========================================================================
    # DataFrame Extraction Section
    # =========================================================================
    print("\nExtracting data to Pandas DataFrame...")
    
    # Get all inputs and outputs from the model
    inputs = prob.model.list_inputs(val=True, out_stream=None)
    outputs = prob.model.list_outputs(val=True, out_stream=None)
    
    # Store them all into a flat dictionary
    all_vars = {}
    for var_name, meta in inputs + outputs:
        val = meta['val']
        
        # Unpack single-element numpy arrays
        if isinstance(val, np.ndarray):
            if val.size == 1:
                val = val[0]
            else:
                continue  # Skip multi-element arrays
                
        # Unpack single-element lists/tuples
        elif isinstance(val, (list, tuple)):
            if len(val) == 1:
                val = val[0]
            else:
                continue  # Skip multi-element lists/tuples
                
        # Only add to the dictionary if it's a true scalar
        if np.isscalar(val) or isinstance(val, (int, float, str, bool)):
            all_vars[var_name] = val
        
    points = ['DESIGN'] + mp_mixedflow.od_pts
    data_rows = []
    
    
    # Reorganize dictionary by mapping point prefixes into rows
    for pt in points:
        pt_dict = {'Point': pt}
        prefix = f"{pt}."
        for var_name, val in all_vars.items():
            if var_name.startswith(prefix):
                param_name = var_name[len(prefix):]  # Strip the prefix to get the column name
                pt_dict[param_name] = val
        data_rows.append(pt_dict)
        
    df = pd.DataFrame(data_rows)
    df.set_index('Point', inplace=True)
    
    # substrings = ['guess', 'balance', 'rhs', 'lhs', 'tp2props', 'resid', 'chem_eq']
    # cols_to_keep = [col for col in df.columns if not any(sub in col for sub in substrings)]
    # df_reduced = df[cols_to_keep]

    important_substrings = ['fc.', 'perf.', 'Fl_I']
    cols_to_keep = [col for col in df.columns if any(sub in col for sub in important_substrings)]
    df_reduced = df[cols_to_keep]

    # Export to CSV for cross-platform debugging/inspection
    csv_filename = "mixedflow_turbofan_30K_080MN_PPL.csv"
    df.to_csv(csv_filename)

    csv_reduced_filename = "mixedflow_turbofan_30K_080MN_PPL_reduced.csv"
    df_reduced.to_csv(csv_reduced_filename)
    
    print(f"\nFull results saved to {csv_filename}")

    # for T in [3200, 3100, 3000]:
    #     prob['balance.rhs:FAR_ab'] = T

    #     prob.run_model()

    #     page_viewer('OD')

    # print()
    # print("time", time.time() - st)