import sys
import pandas as pd
import numpy as np
import openmdao.api as om
import pycycle.api as pyc


class HBTF(pyc.Cycle):

    def initialize(self):
        # Initialize the model here by setting option variables such as a switch for design vs off-des cases
        self.options.declare('throttle_mode', default='T4', values=['T4', 'percent_thrust'])

        super().initialize()


    def setup(self):
        #Setup the problem by including all the relavant components here - comp, burner, turbine etc

        #Create any relavent short hands here:
        design = self.options['design']

        USE_TABULAR = False
        if USE_TABULAR:
            self.options['thermo_method'] = 'TABULAR'
            self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = 'FAR'
        else:
            self.options['thermo_method'] = 'CEA'
            self.options['thermo_data'] = pyc.species_data.janaf
            FUEL_TYPE = 'Jet-A(g)'


        #Add subsystems to build the engine deck:
        self.add_subsystem('Amb', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())

        # Note variable promotion for the fan --
        # the LP spool speed and the fan speed are INPUTS that are promoted:
        # Note here that promotion aliases are used. Here Nmech is being aliased to N1
        # check out: https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_groups/add_subsystem.html?highlight=alias
        self.add_subsystem('fan', pyc.Compressor(map_data=pyc.FanMap,
                                        bleed_names=[], map_extrap=True), promotes_inputs=[('Nmech','N1')])
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct())
        self.add_subsystem('lpc', pyc.Compressor(map_data=pyc.LPCMap,
                                        map_extrap=True),promotes_inputs=[('Nmech','N1')])
        self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem('hpc', pyc.Compressor(map_data=pyc.HPCMap,
                                        bleed_names=['cool1','cool2','cust'], map_extrap=True),promotes_inputs=[('Nmech','N2')])
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['cool3','cool4']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem('hpt', pyc.Turbine(map_data=pyc.HPTMap,
                                        bleed_names=['cool3','cool4'], map_extrap=True),promotes_inputs=[('Nmech','N2')])
        self.add_subsystem('duct11', pyc.Duct())
        self.add_subsystem('lpt', pyc.Turbine(map_data=pyc.LPTMap,
                                        bleed_names=['cool1','cool2'], map_extrap=True),promotes_inputs=[('Nmech','N1')])
        self.add_subsystem('duct13', pyc.Duct())
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct())
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        #Create shaft instances. Note that LP shaft has 3 ports! => no gearbox
        self.add_subsystem('lp_shaft', pyc.Shaft(num_ports=3),promotes_inputs=[('Nmech','N1')])
        self.add_subsystem('hp_shaft', pyc.Shaft(num_ports=2),promotes_inputs=[('Nmech','N2')])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        # Now use the explicit connect method to make connections -- connect(<from>, <to>)

        #Connect the inputs to perf group
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('hpc.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')

        #LP-shaft connections
        self.connect('fan.trq', 'lp_shaft.trq_0')
        self.connect('lpc.trq', 'lp_shaft.trq_1')
        self.connect('lpt.trq', 'lp_shaft.trq_2')
        #HP-shaft connections
        self.connect('hpc.trq', 'hp_shaft.trq_0')
        self.connect('hpt.trq', 'hp_shaft.trq_1')
        #Ideally expanding flow by conneting flight condition static pressure to nozzle exhaust pressure
        self.connect('Amb.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('Amb.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        #Create a balance component
        # Balances can be a bit confusing, here's some explanation -
        #   State Variables:
        #           (W)        Inlet mass flow rate to implictly balance thrust
        #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
        #
        #           (FAR)      Fuel-air ratio to balance Tt4
        #                      LHS: burner.Fl_O:tot:T  == RHS: Tt4 target (set when TF is instantiated)
        #
        #           (lpt_PR)   LPT press ratio to balance shaft power on the low spool
        #           (hpt_PR)   HPT press ratio to balance shaft power on the high spool
        # Ref: look at the XDSM diagrams in the pyCycle paper and this:
        # http://openmdao.org/newdocs/versions/latest/features/building_blocks/components/balance_comp.html

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            balance.add_balance('W', units='lbm/s', eq_units='lbf')
            #Here balance.W is implicit state variable that is the OUTPUT of balance object
            self.connect('balance.W', 'Amb.W') #Connect the output of balance to the relevant input
            self.connect('perf.Fn', 'balance.lhs:W')       #This statement makes perf.Fn the LHS of the balance eqn.
            self.promotes('balance', inputs=[('rhs:W', 'Fn_DES')])

            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=.017)
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
            self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])

            # Note that for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance('lpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.lpt_PR', 'lpt.PR')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

            balance.add_balance('hpt_PR', val=1.5, lower=1.001, upper=8,
                                eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.hpt_PR', 'hpt.PR')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')

        else:

            #In OFF-DESIGN mode we need to redefine the balances:
            #   State Variables:
            #           (W)        Inlet mass flow rate to balance core flow area
            #                      LHS: core_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (FAR)      Fuel-air ratio to balance Thrust req.
            #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
            #
            #           (BPR)      Bypass ratio to balance byp. noz. area
            #                      LHS: byp_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (N1)   LP spool speed to balance shaft power on the low spool
            #           (N2)   HP spool speed to balance shaft power on the high spool

            if self.options['throttle_mode'] == 'T4':
                balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='degR')
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')
                self.promotes('balance', inputs=[('rhs:FAR', 'T4_MAX')])

            elif self.options['throttle_mode'] == 'percent_thrust':
                balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='lbf', use_mult=True)
                self.connect('balance.FAR', 'burner.Fl_I:FAR')
                self.connect('perf.Fn', 'balance.rhs:FAR')
                self.promotes('balance', inputs=[('mult:FAR', 'PC'), ('lhs:FAR', 'Fn_max')])


            balance.add_balance('W', units='lbm/s', lower=10., upper=1000., eq_units='inch**2')
            self.connect('balance.W', 'Amb.W')
            self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

            balance.add_balance('BPR', lower=2., upper=10., eq_units='inch**2')
            self.connect('balance.BPR', 'splitter.BPR')
            self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

            # Again for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance('N1', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.N1', 'N1')
            self.connect('lp_shaft.pwr_in_real', 'balance.lhs:N1')
            self.connect('lp_shaft.pwr_out_real', 'balance.rhs:N1')

            balance.add_balance('N2', val=1.5, units='rpm', lower=500., eq_units='hp', use_mult=True, mult_val=-1)
            self.connect('balance.N2', 'N2')
            self.connect('hp_shaft.pwr_in_real', 'balance.lhs:N2')
            self.connect('hp_shaft.pwr_out_real', 'balance.rhs:N2')

            # Specify the order in which the subsystems are executed:

            # self.set_order(['balance', 'Amb', 'inlet', 'fan', 'splitter', 'duct4', 'lpc', 'duct6', 'hpc', 'bld3', 'burner', 'hpt', 'duct11',
            #                 'lpt', 'duct13', 'core_nozz', 'byp_bld', 'duct15', 'byp_nozz', 'lp_shaft', 'hp_shaft', 'perf'])

        # Set up all the flow connections:
        self.pyc_connect_flow('Amb.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        self.pyc_connect_flow('duct4.Fl_O', 'lpc.Fl_I')
        self.pyc_connect_flow('lpc.Fl_O', 'duct6.Fl_I')
        self.pyc_connect_flow('duct6.Fl_O', 'hpc.Fl_I')
        self.pyc_connect_flow('hpc.Fl_O', 'bld3.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O','core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')

        #Bleed flows:
        self.pyc_connect_flow('hpc.cool1', 'lpt.cool1', connect_stat=False)
        self.pyc_connect_flow('hpc.cool2', 'lpt.cool2', connect_stat=False)
        self.pyc_connect_flow('bld3.cool3', 'hpt.cool3', connect_stat=False)
        self.pyc_connect_flow('bld3.cool4', 'hpt.cool4', connect_stat=False)

        #Specify solver settings:
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8

        # set this very small, so it never activates and we rely on atol
        newton.options['rtol'] = 1e-99
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 1000
        newton.options['reraise_child_analysiserror'] = False
        # ls = newton.linesearch = BoundsEnforceLS()
        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options['maxiter'] = 3
        ls.options['rho'] = 0.75
        # ls.options['print_bound_enforce'] = True

        self.linear_solver = om.DirectSolver()

        super().setup()


def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    if pt == 'DESIGN':
        MN = prob['DESIGN.Amb.Fl_O:stat:MN']
        LPT_PR = prob['DESIGN.balance.lpt_PR']
        HPT_PR = prob['DESIGN.balance.hpt_PR']
        FAR = prob['DESIGN.balance.FAR']
    else:
        MN = prob[pt+'.Amb.Fl_O:stat:MN']
        LPT_PR = prob[pt+'.lpt.PR']
        HPT_PR = prob[pt+'.hpt.PR']
        FAR = prob[pt+'.balance.FAR']

    summary_data = (MN[0], prob[pt+'.Amb.alt'][0], prob[pt+'.inlet.Fl_O:stat:W'][0], prob[pt+'.perf.Fn'][0],
                        prob[pt+'.perf.Fg'][0], prob[pt+'.inlet.F_ram'][0], prob[pt+'.perf.OPR'][0],
                        prob[pt+'.perf.TSFC'][0], prob[pt+'.splitter.BPR'][0])

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" %summary_data, file=file, flush=True)


    fs_names = ['Amb.Fl_O', 'inlet.Fl_O', 'fan.Fl_O', 'splitter.Fl_O1', 'splitter.Fl_O2',
                'duct4.Fl_O', 'lpc.Fl_O', 'duct6.Fl_O', 'hpc.Fl_O', 'bld3.Fl_O', 'burner.Fl_O',
                'hpt.Fl_O', 'duct11.Fl_O', 'lpt.Fl_O', 'duct13.Fl_O', 'core_nozz.Fl_O', 'byp_bld.Fl_O',
                'duct15.Fl_O', 'byp_nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['fan', 'lpc', 'hpc']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f'{pt}.burner'], file=file)

    turb_names = ['hpt', 'lpt']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['core_nozz', 'byp_nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['hp_shaft', 'lp_shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ['hpc', 'bld3', 'byp_bld']
    bleed_full_names = [f'{pt}.{b}' for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


class MPhbtf(pyc.MPCycle):

    def setup(self):

        self.pyc_add_pnt('DESIGN', HBTF(thermo_method='CEA')) # Create an instace of the High Bypass ratio Turbofan

        self.set_input_defaults('DESIGN.inlet.MN', 0.751)
        self.set_input_defaults('DESIGN.fan.MN', 0.4578)
        self.set_input_defaults('DESIGN.splitter.BPR', 5.105)
        self.set_input_defaults('DESIGN.splitter.MN1', 0.3104)
        self.set_input_defaults('DESIGN.splitter.MN2', 0.4518)
        self.set_input_defaults('DESIGN.duct4.MN', 0.3121)
        self.set_input_defaults('DESIGN.lpc.MN', 0.3059)
        self.set_input_defaults('DESIGN.duct6.MN', 0.3563)
        self.set_input_defaults('DESIGN.hpc.MN', 0.2442)
        self.set_input_defaults('DESIGN.bld3.MN', 0.3000)
        self.set_input_defaults('DESIGN.burner.MN', 0.1025)
        self.set_input_defaults('DESIGN.hpt.MN', 0.3650)
        self.set_input_defaults('DESIGN.duct11.MN', 0.3063)
        self.set_input_defaults('DESIGN.lpt.MN', 0.4127)
        self.set_input_defaults('DESIGN.duct13.MN', 0.4463)
        self.set_input_defaults('DESIGN.byp_bld.MN', 0.4489)
        self.set_input_defaults('DESIGN.duct15.MN', 0.4589)
        self.set_input_defaults('DESIGN.N1', 4666.1, units='rpm')
        self.set_input_defaults('DESIGN.N2', 14705.7, units='rpm')

        # --- Set up bleed values -----

        # Define all points to apply parameters to
        all_pts = ['DESIGN', 'OD_full_pwr', 'OD_part_pwr']

        for pt in all_pts:
            self.set_input_defaults(f'{pt}.inlet.ram_recovery', 0.9990)
            self.set_input_defaults(f'{pt}.duct4.dPqP', 0.0048)
            self.set_input_defaults(f'{pt}.duct6.dPqP', 0.0101)
            self.set_input_defaults(f'{pt}.burner.dPqP', 0.0540)
            self.set_input_defaults(f'{pt}.duct11.dPqP', 0.0051)
            self.set_input_defaults(f'{pt}.duct13.dPqP', 0.0107)
            self.set_input_defaults(f'{pt}.duct15.dPqP', 0.0149)
            self.set_input_defaults(f'{pt}.core_nozz.Cv', 0.9933)
            self.set_input_defaults(f'{pt}.byp_bld.bypBld:frac_W', 0.005)
            self.set_input_defaults(f'{pt}.byp_nozz.Cv', 0.9939)
            self.set_input_defaults(f'{pt}.hpc.cool1:frac_W', 0.050708)
            self.set_input_defaults(f'{pt}.hpc.cool1:frac_P', 0.5)
            self.set_input_defaults(f'{pt}.hpc.cool1:frac_work', 0.5)
            self.set_input_defaults(f'{pt}.hpc.cool2:frac_W', 0.020274)
            self.set_input_defaults(f'{pt}.hpc.cool2:frac_P', 0.55)
            self.set_input_defaults(f'{pt}.hpc.cool2:frac_work', 0.5)
            self.set_input_defaults(f'{pt}.bld3.cool3:frac_W', 0.067214)
            self.set_input_defaults(f'{pt}.bld3.cool4:frac_W', 0.101256)
            self.set_input_defaults(f'{pt}.hpc.cust:frac_P', 0.5)
            self.set_input_defaults(f'{pt}.hpc.cust:frac_work', 0.5)
            self.set_input_defaults(f'{pt}.hpc.cust:frac_W', 0.0445)
            self.set_input_defaults(f'{pt}.hpt.cool3:frac_P', 1.0)
            self.set_input_defaults(f'{pt}.hpt.cool4:frac_P', 0.0)
            self.set_input_defaults(f'{pt}.lpt.cool1:frac_P', 1.0)
            self.set_input_defaults(f'{pt}.lpt.cool2:frac_P', 0.0)
            self.set_input_defaults(f'{pt}.hp_shaft.HPX', 250.0, units='hp')

        self.od_pts = ['OD_full_pwr', 'OD_part_pwr']

        self.od_MNs = [0.8, 0.8]
        self.od_alts = [35000.0, 35000.0]
        self.od_Fn_target = [5500.0, 5300]
        self.od_dTs = [0.0, 0.0]

        self.pyc_add_pnt('OD_full_pwr', HBTF(design=False, thermo_method='CEA', throttle_mode='T4'))

        self.set_input_defaults('OD_full_pwr.Amb.MN', 0.8)
        self.set_input_defaults('OD_full_pwr.Amb.alt', 35000, units='ft')
        self.set_input_defaults('OD_full_pwr.Amb.dTs', 0., units='degR')

        self.pyc_add_pnt('OD_part_pwr', HBTF(design=False, thermo_method='CEA', throttle_mode='percent_thrust'))

        self.set_input_defaults('OD_part_pwr.Amb.MN', 0.8)
        self.set_input_defaults('OD_part_pwr.Amb.alt', 35000, units='ft')
        self.set_input_defaults('OD_part_pwr.Amb.dTs', 0., units='degR')

        self.connect('OD_full_pwr.perf.Fn', 'OD_part_pwr.Fn_max')

        self.pyc_use_default_des_od_conns()

        #Set up the RHS of the balances!
        self.pyc_connect_des_od('core_nozz.Throat:stat:area','balance.rhs:W')
        self.pyc_connect_des_od('byp_nozz.Throat:stat:area','balance.rhs:BPR')

        super().setup()

import pandas as pd

def extract_point_data(prob, pt):
    """
    Extracts key cycle parameters from the OpenMDAO problem object 
    into a flat dictionary for DataFrame rows.
    """
    row = {'Point': pt}

    # 1. Handle structural differences in balance variables
    if pt == 'DESIGN':
        row['MN'] = prob.get_val('DESIGN.Amb.Fl_O:stat:MN')[0]
        row['LPT_PR'] = prob.get_val('DESIGN.balance.lpt_PR')[0]
        row['HPT_PR'] = prob.get_val('DESIGN.balance.hpt_PR')[0]
        row['FAR'] = prob.get_val('DESIGN.balance.FAR')[0]
    else:
        row['MN'] = prob.get_val(f'{pt}.Amb.Fl_O:stat:MN')[0]
        row['LPT_PR'] = prob.get_val(f'{pt}.lpt.PR')[0]
        row['HPT_PR'] = prob.get_val(f'{pt}.hpt.PR')[0]
        row['FAR'] = prob.get_val(f'{pt}.balance.FAR')[0]

    # 2. Extract Overall Performance Summary Data
    row['Alt'] = prob.get_val(f'{pt}.Amb.alt')[0]
    row['W'] = prob.get_val(f'{pt}.inlet.Fl_O:stat:W')[0]
    row['Fn'] = prob.get_val(f'{pt}.perf.Fn')[0]
    row['Fg'] = prob.get_val(f'{pt}.perf.Fg')[0]
    row['Fram'] = prob.get_val(f'{pt}.inlet.F_ram')[0]
    row['OPR'] = prob.get_val(f'{pt}.perf.OPR')[0]
    row['TSFC'] = prob.get_val(f'{pt}.perf.TSFC')[0]
    row['BPR'] = prob.get_val(f'{pt}.splitter.BPR')[0]

    # 3. Extract Component Specifics (Compressors)
    comp_names = ['fan', 'lpc', 'hpc']
    for c in comp_names:
        row[f'{c}_PR'] = prob.get_val(f'{pt}.{c}.PR')[0]
        row[f'{c}_eff'] = prob.get_val(f'{pt}.{c}.eff')[0]
        row[f'{c}_Nc'] = prob.get_val(f'{pt}.{c}.Nc')[0]
        row[f'{c}_SMN'] = prob.get_val(f'{pt}.{c}.SMN')[0] # Surge Margin

    # 4. Extract Burner & Turbines
    row['Tt4_Out'] = prob.get_val(f'{pt}.burner.Fl_O:tot:T')[0]
    row['HPT_eff'] = prob.get_val(f'{pt}.hpt.eff')[0]
    row['LPT_eff'] = prob.get_val(f'{pt}.lpt.eff')[0]

    # 5. Extract specific critical flow stations (expand as needed)
    fs_names = ['inlet.Fl_O', 'hpc.Fl_O', 'core_nozz.Fl_O', 'byp_nozz.Fl_O']
    for fs in fs_names:
        row[f'{fs}_Tt'] = prob.get_val(f'{pt}.{fs}:tot:T')[0]
        row[f'{fs}_Pt'] = prob.get_val(f'{pt}.{fs}:tot:P')[0]
        row[f'{fs}_W'] = prob.get_val(f'{pt}.{fs}:stat:W')[0]

    return row
if __name__ == "__main__":
    import time
    from pathlib import Path

    prob = om.Problem()
    prob.model = mp_hbtf = MPhbtf()
    prob.setup()

    # --- [Setup variables and guesses remain unchanged] ---
    prob.set_val('DESIGN.fan.PR', 1.685)
    prob.set_val('DESIGN.fan.eff', 0.8948)
    prob.set_val('DESIGN.lpc.PR', 1.935)
    prob.set_val('DESIGN.lpc.eff', 0.9243)
    prob.set_val('DESIGN.hpc.PR', 9.369)
    prob.set_val('DESIGN.hpc.eff', 0.8707)
    prob.set_val('DESIGN.hpt.eff', 0.8888)
    prob.set_val('DESIGN.lpt.eff', 0.8996)
    prob.set_val('DESIGN.Amb.alt', 35000., units='ft')
    prob.set_val('DESIGN.Amb.MN', 0.8)
    prob.set_val('DESIGN.T4_MAX', 2857, units='degR')
    prob.set_val('DESIGN.Fn_DES', 5900.0, units='lbf')
    prob.set_val('OD_full_pwr.T4_MAX', 2857, units='degR')
    prob.set_val('OD_part_pwr.PC', 0.8)

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.025
    prob['DESIGN.balance.W'] = 100.
    prob['DESIGN.balance.lpt_PR'] = 4.0
    prob['DESIGN.balance.hpt_PR'] = 3.0
    prob['DESIGN.Amb.balance.Pt'] = 5.2
    prob['DESIGN.Amb.balance.Tt'] = 440.0

    for pt in ['OD_full_pwr', 'OD_part_pwr']:
        # initial guesses
        prob[pt+'.balance.FAR'] = 0.02467
        prob[pt+'.balance.W'] = 300
        prob[pt+'.balance.BPR'] = 5.105
        prob[pt+'.balance.N1'] = 5000
        prob[pt+'.balance.N2'] = 15000
        prob[pt+'.hpt.PR'] = 3.
        prob[pt+'.lpt.PR'] = 4.
        prob[pt+'.fan.map.RlineMap'] = 2.0
        prob[pt+'.lpc.map.RlineMap'] = 2.0
        prob[pt+'.hpc.map.RlineMap'] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    flight_env = [(0.00, 0)]
    results_data = []
    first_pass = True

    # Use pathlib for cross-platform paths
    out_file = Path('hbtf_view.out')
    csv_file = Path('turbofan_cycle_data_test.csv')

    # Use a context manager to ensure the file closes properly on both OSs
    with open(out_file, 'w') as viewer_file:
        for MN, alt in flight_env:
            print('***'*10)
            print(f'* MN: {MN}, alt: {alt}')
            print('***'*10)
            
            prob['OD_full_pwr.Amb.MN'] = MN
            prob['OD_full_pwr.Amb.alt'] = alt
            prob['OD_part_pwr.Amb.MN'] = MN
            prob['OD_part_pwr.Amb.alt'] = alt

            # Define the throttle sequence in one list to avoid repeating the execution block
            throttle_sequence = [1, 0.9, 0.8, 0.7, 1, 0.85]

            for PC in throttle_sequence:
                print(f'## PC = {PC}')
                prob['OD_part_pwr.PC'] = PC
                
                try:
                    prob.run_model()
                except om.AnalysisError:
                    print(f"Solver failed to converge at MN={MN}, Alt={alt}, PC={PC}. Skipping.")
                    continue # Skip to the next PC without crashing the whole script

                if first_pass:
                    des_row = extract_point_data(prob, 'DESIGN')
                    des_row['Power_Code'] = 'N/A'
                    results_data.append(des_row)
                    viewer(prob, 'DESIGN', file=viewer_file)
                    first_pass = False
                
                od_row = extract_point_data(prob, 'OD_part_pwr')
                od_row['Power_Code'] = PC
                results_data.append(od_row)
                viewer(prob, 'OD_part_pwr', file=viewer_file)

    # Convert to pandas DataFrame and save
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(csv_file, index=False)
    
    print("\nData extraction complete. First 5 rows:")
    print(df_results.head())
    print(f"\nRun time: {time.time() - st:.2f} seconds")