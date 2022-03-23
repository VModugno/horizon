Search.setIndex({docnames:["docker","horizon","horizon.ros","horizon.solvers","horizon.transcriptions","horizon.utils","index","scheme"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.todo":1,sphinx:55},filenames:["docker.rst","horizon.rst","horizon.ros.rst","horizon.solvers.rst","horizon.transcriptions.rst","horizon.utils.rst","index.rst","scheme.rst"],objects:{"":{horizon:[1,0,0,"-"]},"horizon.functions":{Constraint:[1,1,1,""],CostFunction:[1,1,1,""],Function:[1,1,1,""],FunctionsContainer:[1,1,1,""],ResidualFunction:[1,1,1,""]},"horizon.functions.Constraint":{getBounds:[1,2,1,""],getLowerBounds:[1,2,1,""],getType:[1,2,1,""],getUpperBounds:[1,2,1,""],setBounds:[1,2,1,""],setLowerBounds:[1,2,1,""],setNodes:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.functions.CostFunction":{getType:[1,2,1,""]},"horizon.functions.Function":{deserialize:[1,2,1,""],getDim:[1,2,1,""],getFunction:[1,2,1,""],getImpl:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""],getParameters:[1,2,1,""],getType:[1,2,1,""],getVariables:[1,2,1,""],serialize:[1,2,1,""],setNodes:[1,2,1,""]},"horizon.functions.FunctionsContainer":{addFunction:[1,2,1,""],build:[1,2,1,""],deserialize:[1,2,1,""],getCnstr:[1,2,1,""],getCnstrDim:[1,2,1,""],getCost:[1,2,1,""],getFunction:[1,2,1,""],removeFunction:[1,2,1,""],serialize:[1,2,1,""],setNNodes:[1,2,1,""]},"horizon.functions.ResidualFunction":{getType:[1,2,1,""]},"horizon.misc_function":{checkNodes:[1,3,1,""],checkValueEntry:[1,3,1,""],listOfListFLOATtoINT:[1,3,1,""],unravelElements:[1,3,1,""]},"horizon.problem":{Problem:[1,1,1,""],pickleable:[1,3,1,""]},"horizon.problem.Problem":{createConstraint:[1,2,1,""],createCost:[1,2,1,""],createFinalConstraint:[1,2,1,""],createFinalCost:[1,2,1,""],createFinalResidual:[1,2,1,""],createInputVariable:[1,2,1,""],createIntermediateConstraint:[1,2,1,""],createIntermediateCost:[1,2,1,""],createIntermediateResidual:[1,2,1,""],createParameter:[1,2,1,""],createResidual:[1,2,1,""],createSingleParameter:[1,2,1,""],createSingleVariable:[1,2,1,""],createStateVariable:[1,2,1,""],createVariable:[1,2,1,""],deserialize:[1,2,1,""],evalFun:[1,2,1,""],getConstraints:[1,2,1,""],getCosts:[1,2,1,""],getDt:[1,2,1,""],getDynamics:[1,2,1,""],getInitialState:[1,2,1,""],getInput:[1,2,1,""],getNNodes:[1,2,1,""],getParameters:[1,2,1,""],getState:[1,2,1,""],getVariables:[1,2,1,""],removeConstraint:[1,2,1,""],removeCostFunction:[1,2,1,""],removeVariable:[1,2,1,""],resetDynamics:[1,2,1,""],save:[1,2,1,""],scopeNodeConstraints:[1,2,1,""],scopeNodeCostFunctions:[1,2,1,""],scopeNodeVars:[1,2,1,""],serialize:[1,2,1,""],setDt:[1,2,1,""],setDynamics:[1,2,1,""],setInitialState:[1,2,1,""],setNNodes:[1,2,1,""],toParameter:[1,2,1,""]},"horizon.ros":{replay_trajectory:[2,0,0,"-"],tf_broadcaster_simple:[2,0,0,"-"],utils:[2,0,0,"-"]},"horizon.ros.replay_trajectory":{normalize_quaternion:[2,3,1,""],replay_trajectory:[2,1,1,""]},"horizon.ros.replay_trajectory.replay_trajectory":{publishContactForces:[2,2,1,""],publish_joints:[2,2,1,""],replay:[2,2,1,""],setSlowDownFactor:[2,2,1,""],sleep:[2,2,1,""]},"horizon.ros.tf_broadcaster_simple":{TransformBroadcaster:[2,1,1,""]},"horizon.ros.tf_broadcaster_simple.TransformBroadcaster":{sendTransform:[2,2,1,""]},"horizon.ros.utils":{roslaunch:[2,3,1,""]},"horizon.solvers":{blocksqp:[3,0,0,"-"],ilqr:[3,0,0,"-"],ipopt:[3,0,0,"-"],nlpsol:[3,0,0,"-"],solver:[3,0,0,"-"],sqp:[3,0,0,"-"]},"horizon.solvers.blocksqp":{BlockSqpSolver:[3,1,1,""]},"horizon.solvers.blocksqp.BlockSqpSolver":{configure_rti:[3,2,1,""]},"horizon.solvers.ilqr":{SolverILQR:[3,1,1,""]},"horizon.solvers.ilqr.SolverILQR":{configure_rti:[3,2,1,""],getDt:[3,2,1,""],getSolutionDict:[3,2,1,""],print_timings:[3,2,1,""],save:[3,2,1,""],set_iteration_callback:[3,2,1,""],solve:[3,2,1,""]},"horizon.solvers.ipopt":{IpoptSolver:[3,1,1,""]},"horizon.solvers.nlpsol":{NlpsolSolver:[3,1,1,""]},"horizon.solvers.nlpsol.NlpsolSolver":{build:[3,2,1,""],getConstraintSolutionDict:[3,2,1,""],getDt:[3,2,1,""],getSolutionDict:[3,2,1,""],solve:[3,2,1,""]},"horizon.solvers.solver":{Solver:[3,1,1,""]},"horizon.solvers.solver.Solver":{configure_rti:[3,2,1,""],getDt:[3,2,1,""],getSolutionDict:[3,2,1,""],make_solver:[3,4,1,""],solve:[3,2,1,""]},"horizon.solvers.sqp":{GNSQPSolver:[3,1,1,""]},"horizon.solvers.sqp.GNSQPSolver":{getAlpha:[3,2,1,""],getBeta:[3,2,1,""],getConstraintNormIterations:[3,2,1,""],getConstraintSolutionDict:[3,2,1,""],getDt:[3,2,1,""],getHessianComputationTime:[3,2,1,""],getLineSearchComputationTime:[3,2,1,""],getObjectiveIterations:[3,2,1,""],getQPComputationTime:[3,2,1,""],getSolutionDict:[3,2,1,""],setAlphaMin:[3,2,1,""],setBeta:[3,2,1,""],set_iteration_callback:[3,2,1,""],solve:[3,2,1,""]},"horizon.transcriptions":{integrators:[4,0,0,"-"],methods:[4,0,0,"-"],transcriptor:[4,0,0,"-"],trial_integrator:[4,0,0,"-"]},"horizon.transcriptions.integrators":{EULER:[4,3,1,""],LEAPFROG:[4,3,1,""],RK2:[4,3,1,""],RK4:[4,3,1,""]},"horizon.transcriptions.methods":{DirectCollocation:[4,1,1,""],MultipleShooting:[4,1,1,""]},"horizon.transcriptions.methods.MultipleShooting":{setDefaultIntegrator:[4,2,1,""]},"horizon.transcriptions.transcriptor":{Transcriptor:[4,1,1,""]},"horizon.transcriptions.transcriptor.Transcriptor":{make_method:[4,4,1,""]},"horizon.transcriptions.trial_integrator":{RK4:[4,3,1,""]},"horizon.type_doc":{BoundsDict:[1,1,1,""]},"horizon.utils":{collision:[5,0,0,"-"],kin_dyn:[5,0,0,"-"],mat_storer:[5,0,0,"-"],plotter:[5,0,0,"-"],refiner:[5,0,0,"-"],resampler_trajectory:[5,0,0,"-"],rti:[5,0,0,"-"],utils:[5,0,0,"-"]},"horizon.utils.collision":{CollisionHandler:[5,1,1,""]},"horizon.utils.collision.CollisionHandler":{clamp:[5,4,1,""],collision_to_capsule:[5,4,1,""],compute_distances:[5,2,1,""],dist_capsule_capsule:[5,4,1,""],dist_segment_segment:[5,4,1,""],get_function:[5,2,1,""]},"horizon.utils.kin_dyn":{ForwardDynamics:[5,1,1,""],InverseDynamics:[5,1,1,""],InverseDynamicsMap:[5,1,1,""],linearized_friction_cone:[5,3,1,""],linearized_friction_cone_map:[5,3,1,""],surface_point_contact:[5,3,1,""]},"horizon.utils.kin_dyn.ForwardDynamics":{call:[5,2,1,""]},"horizon.utils.kin_dyn.InverseDynamics":{call:[5,2,1,""]},"horizon.utils.kin_dyn.InverseDynamicsMap":{call:[5,2,1,""]},"horizon.utils.mat_storer":{matStorer:[5,1,1,""],matStorerIO:[5,1,1,""],setInitialGuess:[5,3,1,""]},"horizon.utils.mat_storer.matStorer":{append:[5,2,1,""],load:[5,2,1,""],save:[5,2,1,""],store:[5,2,1,""]},"horizon.utils.mat_storer.matStorerIO":{append:[5,2,1,""],argParse:[5,2,1,""],load:[5,2,1,""],save:[5,2,1,""],store:[5,2,1,""]},"horizon.utils.plotter":{PlotterHorizon:[5,1,1,""]},"horizon.utils.plotter.PlotterHorizon":{plotFunction:[5,2,1,""],plotFunctions:[5,2,1,""],plotVariable:[5,2,1,""],plotVariables:[5,2,1,""],setSolution:[5,2,1,""]},"horizon.utils.refiner":{Refiner:[5,1,1,""]},"horizon.utils.refiner.Refiner":{addProximalCosts:[5,2,1,""],expandDt:[5,2,1,""],expand_nodes:[5,2,1,""],find_nodes_to_inject:[5,2,1,""],getAugmentedProblem:[5,2,1,""],getSolution:[5,2,1,""],get_node_time:[5,2,1,""],group_elements:[5,2,1,""],resetFunctions:[5,2,1,""],resetInitialGuess:[5,2,1,""],resetProblem:[5,2,1,""],resetVarBounds:[5,2,1,""],solveProblem:[5,2,1,""]},"horizon.utils.resampler_trajectory":{resample_input:[5,3,1,""],resample_torques:[5,3,1,""],resampler:[5,3,1,""],second_order_resample_integrator:[5,3,1,""]},"horizon.utils.rti":{RealTimeIteration:[5,1,1,""]},"horizon.utils.rti.RealTimeIteration":{integrate:[5,2,1,""],run:[5,2,1,""]},"horizon.utils.utils":{double_integrator:[5,3,1,""],double_integrator_with_floating_base:[5,3,1,""],jac:[5,3,1,""],quaterion_product:[5,3,1,""],skew:[5,3,1,""],toRot:[5,3,1,""]},"horizon.variables":{AbstractAggregate:[1,1,1,""],AbstractVariable:[1,1,1,""],AbstractVariableView:[1,1,1,""],Aggregate:[1,1,1,""],InputAggregate:[1,1,1,""],InputVariable:[1,1,1,""],OffsetAggregate:[1,1,1,""],OffsetParameter:[1,1,1,""],OffsetVariable:[1,1,1,""],Parameter:[1,1,1,""],ParameterView:[1,1,1,""],SingleParameter:[1,1,1,""],SingleParameterView:[1,1,1,""],SingleVariable:[1,1,1,""],SingleVariableView:[1,1,1,""],StateAggregate:[1,1,1,""],StateVariable:[1,1,1,""],Variable:[1,1,1,""],VariableView:[1,1,1,""],VariablesContainer:[1,1,1,""]},"horizon.variables.AbstractAggregate":{getVars:[1,2,1,""]},"horizon.variables.AbstractVariable":{getDim:[1,2,1,""],getName:[1,2,1,""],getOffset:[1,2,1,""]},"horizon.variables.AbstractVariableView":{getName:[1,2,1,""]},"horizon.variables.Aggregate":{addVariable:[1,2,1,""],getBounds:[1,2,1,""],getInitialGuess:[1,2,1,""],getLowerBounds:[1,2,1,""],getUpperBounds:[1,2,1,""],getVarIndex:[1,2,1,""],getVarOffset:[1,2,1,""],removeVariable:[1,2,1,""],setBounds:[1,2,1,""],setInitialGuess:[1,2,1,""],setLowerBounds:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.variables.OffsetAggregate":{getVarIndex:[1,2,1,""]},"horizon.variables.OffsetParameter":{getImpl:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""]},"horizon.variables.OffsetVariable":{getImpl:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""]},"horizon.variables.Parameter":{assign:[1,2,1,""],getImpl:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""],getParOffset:[1,2,1,""],getParOffsetDict:[1,2,1,""],getValues:[1,2,1,""]},"horizon.variables.ParameterView":{assign:[1,2,1,""],getValues:[1,2,1,""]},"horizon.variables.SingleParameter":{assign:[1,2,1,""],getImpl:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""],getParOffset:[1,2,1,""],getParOffsetDict:[1,2,1,""],getValues:[1,2,1,""]},"horizon.variables.SingleParameterView":{assign:[1,2,1,""]},"horizon.variables.SingleVariable":{getBounds:[1,2,1,""],getImpl:[1,2,1,""],getImplDim:[1,2,1,""],getInitialGuess:[1,2,1,""],getLowerBounds:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""],getUpperBounds:[1,2,1,""],getVarOffset:[1,2,1,""],getVarOffsetDict:[1,2,1,""],setBounds:[1,2,1,""],setInitialGuess:[1,2,1,""],setLowerBounds:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.variables.SingleVariableView":{setBounds:[1,2,1,""],setInitialGuess:[1,2,1,""],setLowerBounds:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.variables.Variable":{getBounds:[1,2,1,""],getImpl:[1,2,1,""],getImplDim:[1,2,1,""],getInitialGuess:[1,2,1,""],getLowerBounds:[1,2,1,""],getName:[1,2,1,""],getNodes:[1,2,1,""],getUpperBounds:[1,2,1,""],getVarOffset:[1,2,1,""],getVarOffsetDict:[1,2,1,""],setBounds:[1,2,1,""],setInitialGuess:[1,2,1,""],setLowerBounds:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.variables.VariableView":{setBounds:[1,2,1,""],setInitialGuess:[1,2,1,""],setLowerBounds:[1,2,1,""],setUpperBounds:[1,2,1,""]},"horizon.variables.VariablesContainer":{createVar:[1,2,1,""],deserialize:[1,2,1,""],getInputVars:[1,2,1,""],getPar:[1,2,1,""],getParList:[1,2,1,""],getStateVars:[1,2,1,""],getVar:[1,2,1,""],getVarList:[1,2,1,""],removeVar:[1,2,1,""],serialize:[1,2,1,""],setInputVar:[1,2,1,""],setNNodes:[1,2,1,""],setParameter:[1,2,1,""],setSingleParameter:[1,2,1,""],setSingleVar:[1,2,1,""],setStateVar:[1,2,1,""],setVar:[1,2,1,""]},horizon:{functions:[1,0,0,"-"],misc_function:[1,0,0,"-"],problem:[1,0,0,"-"],ros:[2,0,0,"-"],solvers:[3,0,0,"-"],transcriptions:[4,0,0,"-"],type_doc:[1,0,0,"-"],utils:[5,0,0,"-"],variables:[1,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","classmethod","Python class method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:function","4":"py:classmethod"},terms:{"0x7f5b6043cfd0":[],"0x7f5b60442070":[],"0x7f5b604c6670":[],"0x7f5b605bcbe0":[],"0x7f5b607b9d60":[],"0x7f63294bd7c0":1,"0x7f63294dc6a0":1,"0x7f63298ca520":1,"0x7f63298e6d30":1,"0x7f6329a691f0":1,"0x7f974bdb8f70":[],"0x7f974bdb8fd0":[],"0x7f974befe640":[],"0x7f974c179d30":[],"0x7f974c1e7b20":[],"0x7fbceb7a6040":[],"0x7fbceb7a60a0":[],"0x7fbceb817b80":[],"0x7fbceb8fd6a0":[],"0x7fbceba18d90":[],"0x7ff14c6f8e20":[],"0x7ff14c6f8e80":[],"0x7ff14c83e520":[],"0x7ff14ca63be0":[],"0x7ff14cb269a0":[],"2nd":4,"4th":4,"abstract":[1,3,7],"case":0,"class":[1,2,3,4,5],"default":[1,3,4,5],"final":[1,4,7],"float":[3,4,5],"function":[4,5,6],"import":[1,4],"int":[1,4],"new":[1,5],"return":[1,3,4,5],"static":0,"true":[1,2,5],"try":[0,6],"var":1,"while":5,FOR:1,The:[0,1,3,4,6,7],Used:1,_mockobject:[1,4],a_r:5,abc:[1,3,4],abstr:1,abstractaggreg:1,abstractvari:1,abstractvariableview:1,acceler:5,accord:5,account:4,act:[5,7],action:[0,3],activ:1,active_nod:1,actual:[4,7],add:[1,4,5],added:1,addfunct:1,addproximalcost:5,addvari:1,after:1,afterward:1,aggreg:1,aim:3,algorithm:6,all:[1,4],allow:[1,6,7],along:[1,4],alpha_min:3,also:[1,4],alwai:1,angular:5,api:6,append:5,appli:1,appreci:0,appropri:3,approxim:4,aquisit:6,arg:1,argpars:5,around:0,arrai:[1,3,5],art:6,articl:0,assign:1,associ:5,augment:7,autodoc:[1,4],automat:1,auxiliari:4,avail:[0,6],base:[1,2,3,4,5,6],base_link:5,befor:1,belong:1,below:6,besid:6,beta:3,better:0,between:[2,4,5],blocksqp:[1,6],blocksqpsolv:3,bool:[1,3],bound:1,boundsdict:1,browser:[0,6],build:[1,3],built:[1,5],call:[3,5],can:[0,1,4,6,7],cannot:0,capabl:[0,6],capsule_1:5,capsule_2:5,cart_pol:0,casadi:[1,5,6,7],casadi_horizon:6,casadi_kin_dyn:5,casadi_typ:4,centauro:6,check:5,checknod:1,checkvalueentri:1,child_frame_id:2,choos:1,clamp:5,classmethod:[3,4,5],clip:6,clone:0,cmth:4,code:[0,1],collect:[1,6],collis:[1,6],collision_to_capsul:5,collisionhandl:5,colloc:4,combin:7,command:0,comment:0,complet:6,compon:[5,7],compphi:4,comput:5,compute_dist:5,concret:3,conda:6,cone:5,configur:6,configure_rti:3,consid:[1,5],constant:5,constrain:[4,5],constraint:[1,4,5,7],construct:[3,4,5],contact:5,contact_fram:5,contain:[1,3,4,5,7],content:6,control:[1,4,5,6],convert:4,coordin:5,core:7,cost:[1,7],costfunct:1,could:4,crash_if_suboptim:1,creat:[1,5],createconstraint:1,createcost:1,createfinalconstraint:1,createfinalcost:1,createfinalresidu:1,createinputvari:1,createintermediateconstraint:1,createintermediatecost:1,createintermediateresidu:1,createparamet:1,createresidu:1,createsingleparamet:1,createsinglevari:1,createstatevari:1,createvar:1,createvari:1,custom:6,customiz:6,dae:[4,5],decis:[1,4],deerial:1,defect:4,defin:[1,4,5],definit:1,degre:4,demo:6,demonstr:[0,6],depend:[3,4],deploy:6,deriv:[1,4,5],describ:[0,1,6],descript:[1,5],deseri:1,desir:[1,4],desired_dt:5,detail:0,dict:[1,3,4,5],dict_valu:5,dictionari:[1,3,4,5],differ:[4,5,6],differenti:6,dim:[1,5],dimens:1,dimension:4,direct:[4,6],directcolloc:4,discret:[3,4],dist_capsule_capsul:5,dist_segment_seg:5,distribut:6,divid:1,docker:6,dof:6,domain:7,don:0,done:5,double_integr:5,double_integrator_with_floating_bas:5,down:2,download:0,dt_rk:4,dummy_nod:1,dure:5,dynam:[1,5,7],each:[1,3],eas:6,element:1,empti:1,engin:7,eras:1,euler:4,evalfun:1,evalu:[1,5],evinron:6,evolut:4,exampl:[0,1,5,6],except:1,execut:[],expand_nod:5,expanddt:5,express:[5,7],ext:[1,4],extra:5,f_rk:4,factor:2,factori:3,fals:[1,5],fator:2,featur:[],ff_r_cf:5,file:2,file_nam:5,fill:3,find:[0,3],find_nodes_to_inject:5,flag:3,follow:[0,1,5],forc:5,force_reference_fram:[2,5],forward:5,forwarddynam:5,found:[1,6],frame:5,frame_force_map:[2,5],frame_id:2,frame_res_force_map:5,framework:[1,6],francesco_ruscelli:6,francescoruscelli:0,friciton:5,from:[0,1,3,5,6],full:6,fun:1,fun_nam:1,function_string_list:5,functionscontain:1,gather:[5,6],gener:[0,1,3,5],get:[1,3],get_funct:5,get_node_tim:5,getalpha:3,getaugmentedproblem:5,getbeta:3,getbound:1,getcnstr:1,getcnstrdim:1,getconstraint:1,getconstraintnormiter:3,getconstraintsolutiondict:3,getcost:1,getdim:1,getdt:[1,3],getdynam:1,getfunct:1,gethessiancomputationtim:3,getimpl:1,getimpldim:1,getinitialguess:1,getinitialst:1,getinput:1,getinputvar:1,getlinesearchcomputationtim:3,getlowerbound:1,getnam:1,getnnod:1,getnod:1,getobjectiveiter:3,getoffset:1,getpar:1,getparamet:1,getparlist:1,getparoffset:1,getparoffsetdict:1,getqpcomputationtim:3,getsolut:5,getsolutiondict:3,getstat:1,getstatevar:1,getter:1,gettyp:1,getupperbound:1,getvalu:1,getvar:1,getvari:1,getvarindex:1,getvarlist:1,getvaroffset:1,getvaroffsetdict:1,given:[1,3,4,5],global:5,gnsqp:6,gnsqpsolver:3,greatli:1,grid:5,group_el:5,guess:1,hand:0,have:0,hello:[],here:[0,6],hessian:5,hide:[],highli:6,horizon_dock:[],html:4,http:4,iii:3,ilqr:[1,6],implement:[0,1,3,4,5,7],includ:0,independ:[1,6],index:6,indic:[1,3],industri:6,infinit:4,initi:1,input:[1,4,5],input_r:5,input_vec:5,inputaggreg:1,inputvari:1,insert:1,insid:[0,1],instal:0,instanc:[1,3,4],integr:[1,5,6],integrator_typ:4,interact:0,interfac:3,intern:[1,5],interv:1,intuit:6,invers:5,inversedynam:5,inversedynamicsmap:5,ipopt:[1,6],ipoptsolv:3,is_floating_bas:2,iter:1,its:[1,3,4],itself:1,jac:5,jac_test:5,jacobian:5,joint:5,joint_list:2,jupyterlab:6,kangaroo:6,kei:3,kept:1,kin_dyn:[1,6],kindyn:[2,5],kinemat:7,kutta:4,last:1,launch:2,leap:0,leapfrog:4,lectur:4,leg:6,legend:5,lies:5,linear:[1,5,6],linearized_friction_con:5,linearized_friction_cone_map:5,link:5,list:[0,1,5],listoflist:1,listoflistfloattoint:1,load:[1,5],local:[2,5],local_world_align:5,localhost:[],locat:2,logger:[1,4,5],logging_level:1,lower:1,lsole:5,machin:[0,6],mackinnon:4,main:1,make:3,make_method:4,make_solv:3,manag:[1,7],mani:6,manipul:6,map:5,marker:5,mat_stor:[1,6],matrix:[1,5],matstor:5,matstorerio:5,mechan:1,method:[1,6],misc_funct:6,model:[5,6],modelpars:7,modul:6,more:1,motion:0,multipl:4,multipleshoot:4,multipli:1,must:1,myvar:1,n_node:1,name:[1,3,5],natur:1,nddot:5,ndot:5,need:3,new_nodes_vec:5,nlp:[6,7],nlpsol:[1,6],nlpsolsolv:3,nnone:3,node11:4,node4:4,node:[1,3,5],node_tim:5,nodes_dt:5,nodes_self:1,non:[1,6],none:[1,2,3,4,5],nonlinear:6,normalize_quaternion:2,note:[1,4,5],now:0,number:[1,4,5],numer:7,numpi:[1,5],obj:1,object:[1,2,4,5],ode:[4,5],off:1,offici:0,offset:1,offset_vari:1,offsetaggreg:1,offsetparamet:1,offsetvari:1,omega:5,one:1,onli:[1,5],onlin:0,opt:[3,4,5],optim:[0,1,3,4,6,7],option:[1,3,4],order:[1,4],ordereddict:1,orient:5,origin:5,out:6,over:[1,4],overrid:1,p_0:1,p_1:1,p_fp1:5,p_fp2:5,p_fq1:5,p_fq2:5,p_n:1,p_re:5,packag:6,page:[0,6],pair:1,par_impl:1,param:[1,2,4,5],paramet:[1,3,4,5,7],parameterview:1,parametr:1,parent:1,parent_nam:1,part:5,peopl:4,perform:4,period:5,pickleabl:1,pinocchio:6,pip:6,pipelin:6,plai:0,plan:1,plane:5,plane_dict:5,playlist:6,plone:[],plot:0,plotfunct:5,plotter:[1,6],plotterhorizon:5,plotvari:5,point:[1,5],pointer:4,polynomi:4,portion:1,pos:2,posit:5,possibl:0,prb:[3,4,5],previou:[1,5],print_tim:3,prob:4,problem:[0,3,4,5,6,7],process:4,produc:7,product:5,project:[1,3],prototyp:6,pseudo:7,publish_joint:2,publishcontactforc:2,pull:0,python3:0,q_replai:2,q_rk:4,qddot:5,qddotj:5,qdot:5,qdotj:5,qp_solver_plugin:3,quad:[4,5],quadratur:[4,5],quat:5,quatdot:5,quaterion_product:5,quaternion:5,quick:6,rather:[1,4],reach:0,real:4,realtimeiter:5,reduc:6,ref:[],refer:[1,5],referencefram:[2,5],refin:[1,6],rel:1,relat:7,reli:6,remov:1,removeconstraint:1,removecostfunct:1,removefunct:1,removevar:1,removevari:1,replai:2,replay:[1,6],replay_trajectori:[1,6],replayer_fd:[1,6],replayer_fd_mx:[1,6],replayer_mx:[1,6],repo:0,repres:5,represent:[1,7],requir:3,resampl:5,resample_input:5,resample_torqu:5,resampler_trajectori:[1,6],resetdynam:1,resetfunct:5,resetinitialguess:5,resetproblem:5,resetvarbound:5,residu:1,residualfunct:1,result:[1,7],retriev:[1,3],rk2:4,rk4:[4,5],robot:[0,6,7],ros:[1,6],roslaunch:2,rot:2,rotat:5,rti:[1,3,6],run:[0,2,5,6],rung:4,rviz:0,rvizweb:[],same:1,sampl:5,save:[1,3,5],scheme:[3,4,6],scope:1,scopenodeconstraint:1,scopenodecostfunct:1,scopenodevar:1,search:6,sec:2,second:2,second_order_resample_integr:5,seen:4,segment:4,self:1,send_to_gazebo:[1,6],sendtransform:2,separ:2,sequenc:2,serial:1,set:[1,2],set_iteration_callback:3,setalphamin:3,setbeta:3,setbound:1,setdefaultintegr:4,setdt:1,setdynam:1,setinitialguess:[1,5],setinitialst:1,setinputvar:1,setlowerbound:1,setnnod:1,setnod:1,setparamet:1,setsingleparamet:1,setsinglevar:1,setslowdownfactor:2,setsolut:5,setstatevar:1,setter:1,setup:6,setupperbound:1,setvar:1,sever:6,shift:1,shoot:[1,4],show:[],show_bound:5,simplifi:1,sinc:[1,4],singl:[1,3],singleparamet:1,singleparameterview:1,singlevari:1,singlevariableview:1,skew:5,sleep:2,slow:2,slow_down_factor:2,smoothli:6,solut:[1,3,4,5],solv:[1,3,4,6,7],solveproblem:5,solver:[1,4,5,6,7],solver_plugin:3,solverilqr:3,some:6,space:5,specifi:1,sphinx:[1,4],spoiler:0,spot:[0,6],sqp:[1,6],state:[1,4,5,6],state_var_impl:3,state_vec:5,stateaggreg:1,stateread:5,statevari:1,step:[0,3,4],store:[1,5],str:[1,3,4],strategi:[4,7],string:[1,3],structur:[1,6,7],studi:0,submodul:6,subpackag:6,success:3,suitabl:3,summari:[1,5],support:6,surfac:5,surface_point_contact:5,symbol:[1,7],symmetr:5,system:[0,1,6],t_co:1,tag:1,tailor:6,take:[3,4],talo:6,tau:5,tau_ext:5,tau_r:5,tbd:[],techniqu:4,term:[4,5],test:5,tf_broadcaster_simpl:[1,6],than:4,thank:7,them:1,thi:[0,1,4,5],thread:2,through:6,throughout:1,time:[2,4,5],todo:5,toggl:[],tool:[1,6],toparamet:1,torot:5,torqu:5,total:1,trajectori:[0,1,2,4,6],transcript:[1,6,7],transcriptionmethod:7,transcriptor:[1,6],transformbroadcast:2,translat:7,trial_integr:[1,6],two:[5,6],type:[1,3,4,5],type_doc:6,typic:3,u_opt:[3,5],u_r:5,u_rk:4,uml:7,union:1,unravelel:1,updat:1,upper:1,urdf:5,urdf_parser_pi:5,urdfstr:5,usag:[3,5],used:[1,4,5],used_par:1,used_var:1,useful:1,useless:1,user:1,uses:[0,6],using:[1,5,6],util:[1,6],v_re:5,val:[1,3],valu:[1,3],var_impl:1,var_nam:1,var_slic:1,var_string_list:5,var_typ:1,variabl:[3,4,5,6,7],variablescontain:1,variableview:1,vec:5,vec_to_expand:5,vector:[1,3,5],veloc:5,vis_refiner_glob:[1,6],vis_refiner_loc:[],visual:0,walk:0,warn:1,wdoti:5,wdotx:5,wdotz:5,web:6,webpag:0,well:5,where:[1,2,5,6],which:[0,1,4,5,7],without:0,work:1,workspac:0,world:5,www:4,x0_rk:4,x_0:1,x_1:1,x_n:1,x_opt:3,x_rk:4,xdot:[1,4,5],xmax:5,xmin:5,yield:4,you:[0,6],your:[0,6],zero:1},titles:["Horizon docker","Horizon package","horizon.ros package","horizon.solvers package","horizon.transcriptions package","horizon.utils package","Welcome to Horizon\u2019s documentation!","Horizon scheme"],titleterms:{"function":1,blocksqp:3,collis:5,content:[1,2,3,4,5],docker:0,document:6,featur:6,get:6,horizon:[0,1,2,3,4,5,6,7],ilqr:3,indic:6,instal:6,integr:4,ipopt:3,kin_dyn:5,mat_stor:5,method:4,misc_funct:1,modul:[1,2,3,4,5],nlpsol:3,packag:[1,2,3,4,5],plotter:5,problem:1,refin:5,replay:5,replay_trajectori:2,replayer_fd:5,replayer_fd_mx:5,replayer_mx:5,resampler_trajectori:5,ros:2,rti:5,scheme:7,send_to_gazebo:5,solver:3,sqp:3,start:6,submodul:[1,2,3,4,5],subpackag:1,tabl:6,tf_broadcaster_simpl:2,transcript:4,transcriptor:4,trial_integr:4,type_doc:1,util:[2,5],variabl:1,video:6,vis_refiner_glob:5,vis_refiner_loc:[],welcom:6}})