module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module NL = Numpy.Lax_numpy
module AC = Numpy.Array_creation
module Poly = Numpy.Polynomial
module NN = Nn.Functions
module JU = Numpy.Jnp_util
module Dts = Dtypes
module L = Lax_api
module CD = Custom_derivatives

let c_lanczos_coeffs =
  [|
    0.9999999999998099;
    676.5203681218851;
    -1259.1392167224028;
    771.3234287776531;
    -176.6150291621406;
    12.507343278686905;
    -0.13857109526572012;
    9.984369578019572e-06;
    1.5056327351493116e-07;
  |]

let c_wofz_c =
  [|
    -1.3034426067909105e-12;
    3.741183837373847e-12;
    8.030427700497756e-12;
    -2.154359355749088e-11;
    -5.544244996323793e-11;
    1.165824698850374e-10;
    4.1537441121999766e-10;
    -5.231020211492061e-10;
    -3.2080151339721323e-09;
    8.124886421653591e-10;
    2.3797556530014025e-08;
    2.2930438438915445e-08;
    -1.4813078929137642e-07;
    -4.1840763750512053e-07;
    4.2558331397138446e-07;
    4.401531731283225e-06;
    6.821031944357515e-06;
    -2.1409619201999998e-05;
    -0.0001307544925457942;
    -0.00024532980270038237;
    0.00039259136070109705;
    0.004519541105349309;
    0.0190061557848455;
    0.05730440352983728;
    0.14060716226893755;
    0.2954445107150874;
    0.5460139720639338;
    0.9019254893648001;
    1.345544169234545;
    1.8256696296324815;
    2.2635372999002676;
    2.5722534081245696;
  |]

let c_dawsn_an =
  [|
    1.1368149897175597e-11;
    8.492622676674738e-10;
    1.9443420417555306e-08;
    9.531517412544844e-07;
    3.078283098749132e-06;
    0.00035251336852028877;
    -0.0008501498467244109;
    0.04226182230055466;
    -0.09174803717734524;
    1.0;
  |]

let c_dawsn_ad =
  [|
    2.403720730667626e-11;
    1.488646813684934e-09;
    5.212652810105416e-08;
    1.2725847827318697e-06;
    2.324902498207895e-05;
    0.00032552474182605793;
    0.003488058146571626;
    0.027944853119882897;
    0.15887424196012057;
    0.5749186294893204;
    1.0;
  |]

let c_dawsn_bn =
  [|
    0.5089551564179009;
    -0.24475441814269785;
    0.09415123353035344;
    -0.021871125514203903;
    0.003662076123295692;
    -0.0004232091144603888;
    3.5964130479389666e-05;
    -2.1464035171996896e-06;
    9.100107800763914e-08;
    -2.4027452082825098e-09;
    3.592333854409284e-11;
  |]

let c_dawsn_bd =
  [|
    1.0;
    -0.6318398698733682;
    0.2367067882282487;
    -0.053180636700322324;
    0.008480417185862954;
    -0.0009479967684866654;
    7.810255929445524e-05;
    -4.558751532524426e-06;
    1.8910035811142186e-07;
    -4.913246913319206e-09;
    7.184664032357345e-11;
  |]

let c_dawsn_cn =
  [|
    -0.5905928605347732;
    0.6292352427243688;
    -0.17285897538038814;
    0.016483704782518965;
    -0.0004868276130204627;
  |]

let c_dawsn_cd =
  [|
    1.0;
    -2.698200571975449;
    1.7327079904594784;
    -0.3937085822819395;
    0.03442789240412334;
    -0.0009736552260409412;
  |]

let c_bernoulli_coefs =
  [|
    12.0;
    -720.0;
    30240.0;
    -1209600.0;
    47900160.0;
    -1892437580.3183792;
    74724249600.0;
    -2950130727918.164;
    116467828143500.67;
    -4597978722407473.0;
    1.8152105401943546e+17;
    -7.166165256175667e+18;
    2.82908877253043e+20;
    -1.1168794925000445e+22;
    4.4092635141854666e+23;
    -1.7407074646225822e+25;
  |]

let c_ndtri_p0 =
  [|
    -59.96335010141079;
    98.00107541859997;
    -56.67628574690703;
    13.931260938727968;
    -1.2391658386738125;
  |]

let c_ndtri_q0 =
  [|
    1.0;
    1.9544885833814176;
    4.676279128988815;
    86.36024213908905;
    -225.46268785411937;
    200.26021238006066;
    -82.03722561683334;
    15.90562251262117;
    -1.1833162112133;
  |]

let c_ndtri_p1 =
  [|
    4.0554489230596245;
    31.525109459989388;
    57.16281922464213;
    44.08050738932008;
    14.684956192885803;
    2.1866330685079025;
    -0.1402560791713545;
    -0.03504246268278482;
    -0.0008574567851546854;
  |]

let c_ndtri_q1 =
  [|
    1.0;
    15.779988325646675;
    45.39076351288792;
    41.3172038254672;
    15.04253856929075;
    2.504649462083094;
    -0.14218292285478779;
    -0.03808064076915783;
    -0.0009332594808954574;
  |]

let c_ndtri_p2 =
  [|
    3.2377489177694603;
    6.915228890689842;
    3.9388102529247444;
    1.3330346081580755;
    0.20148538954917908;
    0.012371663481782003;
    0.00030158155350823543;
    2.6580697468673755e-06;
    6.239745391849833e-09;
  |]

let c_ndtri_q2 =
  [|
    1.0;
    6.02427039364742;
    3.6798356385616087;
    1.3770209948908132;
    0.21623699359449663;
    0.013420400608854318;
    0.00032801446468212774;
    2.8924786474538068e-06;
    6.790194080099813e-09;
  |]

let c_expint1_a =
  [|
    -5.350447357812543;
    218.50491688166133;
    -4176.572384826693;
    55411.76756393557;
    -331338.13311781443;
    1592627.1633849454;
  |]

let c_expint1_b =
  [|
    1.0;
    -52.505479591128626;
    1259.6161867867906;
    -17565.495819735348;
    149306.2117002726;
    -729494.9239640528;
    1592627.1633849454;
  |]

let c_expint2_a =
  [|
    0.019818085032596897;
    -1.2716456259849176;
    -2.0881603356812284;
    2.7555445091879367;
    -0.44095070487016;
    0.046656238059358916;
    -0.0015450426796734852;
    7.059980605299618e-05;
  |]

let c_expint2_b =
  [|
    1.0;
    1.4764986709149215;
    0.5629177174822436;
    0.16990178978793072;
    0.02291647179034212;
    0.0044501504397287525;
    0.00017274396122065214;
    3.953167195549672e-05;
  |]

let c_expint3_a =
  [|
    -1.3732153758712087;
    -0.7084559133740839;
    1.580806855547941;
    -0.2601500427425623;
    0.029946746941137137;
    -0.001038086040188744;
    4.371064420753005e-05;
    2.141783679522603e-06;
  |]

let c_expint3_b =
  [|
    1.0;
    0.8585231423622028;
    0.4483285822873995;
    0.07687932158124476;
    0.024498682410218877;
    0.0008832165941927797;
    0.00045909522995113536;
    -4.729848351866523e-06;
    2.66519553739071e-06;
  |]

let c_expint4_a =
  [|
    -2.1069346016919166;
    1.7327338696646881;
    -0.2423619178935842;
    0.02322724180937566;
    0.00023728804404931798;
    -8.343219561192553e-05;
    1.3634087956052505e-05;
    -3.655412321999254e-07;
    1.4649417339759612e-08;
    6.17640786371036e-10;
  |]

let c_expint4_b =
  [|
    1.0;
    -0.2298062239901678;
    0.11050770414740378;
    -0.015665429666307924;
    0.002761106850817353;
    -0.00020891480122840484;
    1.7085289388076754e-05;
    -4.4593117963566865e-07;
    1.3946349303538475e-08;
    6.150865933977338e-10;
  |]

let c_expint5_a =
  [|
    -0.24581193676740204;
    -0.14833822533220778;
    0.07248291795735552;
    -0.013483156873809405;
    0.001342775069788637;
    -7.942465637159712e-05;
    2.644179518984236e-06;
    -4.239473659313765e-08;
  |]

let c_expint5_b =
  [|
    1.0;
    -0.10442259084438711;
    -0.26764531281014026;
    0.09695000254621984;
    -0.01601745692712991;
    0.001496414899205908;
    -8.462452563778485e-05;
    2.7289384034767266e-06;
    -4.239462431819542e-08;
  |]

let c_expint6_a =
  [|
    0.12125611181054567;
    -0.5823133179043894;
    0.2348887314557017;
    -0.030400343181132484;
    0.0015100821468651906;
    -2.5231370954995714e-05;
  |]

let c_expint6_b =
  [|
    1.0;
    -1.002252150365854;
    0.2928709694872224;
    -0.03337004338674008;
    0.0015605448811273888;
    -2.5231370936032347e-05;
  |]

let c_expint7_a =
  [|
    -0.7657847078286127;
    0.6886192415566705;
    -0.2132598113545206;
    0.033461075523841935;
    -0.003076541477344756;
    0.00017471193164549074;
    -6.1037116822741705e-06;
    1.218032765428652e-07;
    -1.0860761027932903e-09;
  |]

let c_expint7_b =
  [|
    1.0;
    -1.8888028686623086;
    1.066691687211409;
    -0.27519159823063805;
    0.039308526882338234;
    -0.003414684558602365;
    0.00018668443707035555;
    -6.345146083130515e-06;
    1.239754287483207e-07;
    -1.0860761027931265e-09;
  |]

let c_sici_series_sn =
  [|
    -8.391678279103039e-11;
    4.625917144270128e-08;
    -9.757593038436328e-06;
    0.0009769454381704354;
    -0.04134703162294066;
    1.0;
  |]

let c_sici_series_sd =
  [|
    2.0326926619595193e-12;
    1.279978911799433e-09;
    4.418278428012189e-07;
    9.964121220438756e-05;
    0.01420852393261499;
    1.0;
  |]

let c_sici_series_cn =
  [|
    2.0252400238910228e-11;
    -1.3524950491579076e-08;
    3.593250514199931e-06;
    -0.0004740072068734079;
    0.028915965260755523;
    -1.0;
  |]

let c_sici_series_cd =
  [|
    4.077460400618806e-12;
    3.067809975818878e-09;
    1.2321035568588342e-06;
    0.00031744202477503275;
    0.051002805623644606;
    4.0;
  |]

let c_sici_asympt_fn4 =
  [|
    4.236128628922166;
    5.4593771716181285;
    1.6208328770153833;
    0.16700661183132304;
    0.006810201324725182;
    0.00010893658065032867;
    5.489002234213736e-07;
  |]

let c_sici_asympt_fd4 =
  [|
    1.0;
    8.16496634205391;
    7.308288225055645;
    1.867922579501842;
    0.1787920529631499;
    0.007017106683227897;
    0.00011003435715391573;
    5.489002527562557e-07;
  |]

let c_sici_asympt_gn4 =
  [|
    0.08710016989731142;
    0.6113791099522193;
    0.3971802963923375;
    0.07485277376284691;
    0.005388686814621773;
    0.00016199979459893403;
    1.9796387414096365e-06;
    7.825790407440903e-09;
  |]

let c_sici_asympt_gd4 =
  [|
    1.0;
    1.6440220241335535;
    0.666296701268988;
    0.09887717612776888;
    0.006223963454417684;
    0.0001732210814741771;
    2.0265918208634397e-06;
    7.825792189335346e-09;
  |]

let c_sici_asympt_fn8 =
  [|
    0.4558808734704653;
    0.7137152741001467;
    0.16030015822231947;
    0.01160642294081244;
    0.00034955644244785906;
    4.8621543082645475e-06;
    3.200927900910049e-08;
    9.41779576128513e-11;
    9.70507110881952e-14;
  |]

let c_sici_asympt_fd8 =
  [|
    1.0;
    0.9174636118736841;
    0.17868554533207454;
    0.012225359477197129;
    0.00035869648188185157;
    4.924350643178815e-06;
    3.21956939101046e-08;
    9.437205903502767e-11;
    9.70507110881952e-14;
  |]

let c_sici_asympt_gn8 =
  [|
    0.6973599534432762;
    0.33041097930563207;
    0.03848787676499743;
    0.001717182390523479;
    3.4894116550227946e-05;
    3.471311670841167e-07;
    1.7040445278204452e-09;
    3.859459254302766e-12;
    3.1404009894636335e-15;
  |]

let c_sici_asympt_gd8 =
  [|
    1.0;
    1.6854889881101165;
    0.48785225869530496;
    0.04679131942596258;
    0.0019028442667439953;
    3.684755044425611e-05;
    3.5704322344374083e-07;
    1.7269374896631615e-09;
    3.878301660239547e-12;
    3.1404009894636335e-15;
  |]

let c_spence_poly_a =
  [|
    4.6512858607399003e-05;
    0.007315890452380947;
    0.13384763957830903;
    0.8796913117545303;
    2.7114985119655346;
    4.256971560081218;
    3.297713409852251;
    1.0;
  |]

let c_spence_poly_b =
  [|
    0.0006909904889125533;
    0.02540437639325444;
    0.2829748606025681;
    1.4117259775183106;
    3.6380053334513707;
    5.03278880143317;
    3.547713409852251;
    1.0;
  |]

let c_bernoulli_b3 = [| 1.0; -0.5; 0.16666666666666666 |]

let c_owens_t_quad_pts =
  [|
    0.0035082039676451716;
    0.031279042338030756;
    0.08526682628321945;
    0.16245071730812277;
    0.25851196049125436;
    0.3680755384069753;
    0.485010929056047;
    0.6027751415261857;
    0.7147788421775323;
    0.814755109887601;
    0.8971102975594897;
    0.9572380808594426;
    0.991788329746297;
  |]

let c_owens_t_quad_wts =
  [|
    0.018831438115323503;
    0.01856708624397765;
    0.018042093461223385;
    0.017263829606398752;
    0.016243219975989858;
    0.014994592034116705;
    0.013535474469662087;
    0.011886351605820165;
    0.010070377242777432;
    0.008113054574229958;
    0.006041900952847024;
    0.0038862217010742057;
    0.001679303108454609;
  |]

let c_fresnl_sn_f32 =
  [|
    1.6476294637887e-09;
    -1.522754752581096e-07;
    8.4247488085024e-06;
    -0.0003120693124703272;
    0.007244727626597022;
    -0.09228055941124598;
    0.5235987735681432;
  |]

let c_fresnl_cn_f32 =
  [|
    1.416802502367354e-08;
    -1.157231412229871e-06;
    5.387223446683264e-05;
    -0.001604381798862293;
    0.02818489036795073;
    -0.2467398198317899;
    0.9999999760004487;
  |]

let c_fresnl_fn_f32 =
  [|
    -1903009855649.792;
    135594238805.0252;
    -4158143148.511033;
    73438484.63587323;
    -873235.6681548485;
    8560.51546627547;
    -103.2877601091159;
    2.999401847870011;
  |]

let c_fresnl_gn_f32 =
  [|
    -186084399762.465;
    12783506733.93208;
    -377938771.3202229;
    6492611.570598858;
    -77877.89623358162;
    860.2931494734327;
    -14.93439396592284;
    0.9999841934744914;
  |]

let c_fresnl_sn_f64 =
  [|
    -2991.8191940101983;
    708840.0452577386;
    -62974148.62058625;
    2548908805.7337637;
    -44297951805.96978;
    318016297876.5678;
  |]

let c_fresnl_sd_f64 =
  [|
    1.0;
    281.3762688899943;
    45584.78108065326;
    5173438.887700964;
    419320245.8981112;
    22441179564.534092;
    607366389490.0846;
  |]

let c_fresnl_cn_f64 =
  [|
    -4.9884311457357354e-08;
    9.504280628298596e-06;
    -0.0006451914356839651;
    0.018884331939670384;
    -0.20552590095501388;
    1.0;
  |]

let c_fresnl_cd_f64 =
  [|
    3.99982968972496e-12;
    9.154392157746574e-10;
    1.2500186247959882e-07;
    1.2226278902417902e-05;
    0.0008680295429417843;
    0.04121420907221998;
    1.0;
  |]

let c_fresnl_fn_f64 =
  [|
    0.4215435550436775;
    0.1434079197807589;
    0.011522095507358577;
    0.000345017939782574;
    4.6361374928786735e-06;
    3.055689837902576e-08;
    1.0230451416490724e-10;
    1.7201074326816183e-13;
    1.3428327623306275e-16;
    3.763297112699879e-20;
  |]

let c_fresnl_fd_f64 =
  [|
    1.0;
    0.7515863983533789;
    0.11688892585919138;
    0.0064405152650885865;
    0.00015593440916415301;
    1.8462756734893055e-06;
    1.1269922476399903e-08;
    3.6014002958937136e-11;
    5.887545336215784e-14;
    4.5200143407412973e-17;
    1.2544323709001127e-20;
  |]

let c_fresnl_gn_f64 =
  [|
    0.5044420736433832;
    0.1971028335255234;
    0.018764858409257526;
    0.0006840793809153931;
    1.1513882611188428e-05;
    9.828524436884223e-08;
    4.4534441586175015e-10;
    1.0826804113902088e-12;
    1.375554606332618e-15;
    8.363544356306774e-19;
    1.8695871016278324e-22;
  |]

let c_fresnl_gd_f64 =
  [|
    1.0;
    1.4749575992512833;
    0.33774898912002;
    0.02536037414203388;
    0.0008146791071843061;
    1.2754507566772912e-05;
    1.0431458965757199e-07;
    4.6068072814652043e-10;
    1.1027321506624028e-12;
    1.3879653125957886e-15;
    8.391588162831187e-19;
    1.8695871016278324e-22;
  |]

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let bind1 = C.bind1
let numel sh = Array.fold_left ( * ) 1 sh
let const_full dt sh x = Core.splat dt sh x

let scalar_of v x =
  let literal = Nd.of_floats (dtype v) [||] [| x |] in
  T.Concrete
    (if (C.get_aval v).weak_type then Nd.with_py_value (Nd.as_weak literal) x
     else literal)

let lax_const v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])
let coeff_of v a = T.Concrete (Nd.of_floats (dtype v) [| Array.length a |] a)
let where_ = NL.where_
let q32 v = Int32.float_of_bits (Int32.bits_of_float v)
let py_float = JU.py_float
let where_px c v y = JU.where_weak ~x_weak:true c (py_float v) y
let where_py c x v = JU.where_weak ~y_weak:true c x (py_float v)
let ( +! ) = UF.add
let ( -! ) = UF.subtract
let ( *! ) = UF.multiply
let ( /! ) = UF.true_divide
let band = UF.bitwise_and
let bor = UF.bitwise_or
let lt_ x v = UF.less x (scalar_of x v)
let le_ x v = UF.less_equal x (scalar_of x v)
let gt_ x v = UF.greater x (scalar_of x v)
let ge_ x v = UF.greater_equal x (scalar_of x v)
let eq_ x v = UF.equal x (scalar_of x v)
let ne_ x v = UF.not_equal x (scalar_of x v)
let neg_ = UF.negative
let lit_add v x = scalar_of x v +! x
let add_lit x v = x +! scalar_of x v
let sub_lit x v = x -! scalar_of x v
let lit_sub v x = scalar_of x v -! x
let lit_mul v x = scalar_of x v *! x
let mul_lit x v = x *! scalar_of x v
let lit_div v x = scalar_of x v /! x
let div_lit x v = x /! scalar_of x v
let polyval = Poly.polyval
let int_full dt v = T.Concrete (Nd.of_floats dt [||] [| float_of_int v |])
let default_int () = Dts.default_int_dtype ()
let py_int = JU.py_int

let to_inexact = function
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
  | ( D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
    | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
    | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Complex64 | D.Complex128 ) as
    d ->
      d

let promote_inexact vs =
  let dt, is_weak =
    Dtypes.result_type
      (List.map (fun v -> (dtype v, (get_aval v).T.weak_type)) vs)
  in
  let dt = to_inexact dt in
  UF.promote_shapes (List.map (fun v -> C.convert_like ~weak:is_weak v dt) vs)

let promote1 x =
  match promote_inexact [ x ] with [ y ] -> y | _ -> assert false

let promote2 a b =
  match promote_inexact [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let promote3 a b c =
  match promote_inexact [ a; b; c ] with
  | [ a; b; c ] -> (a, b, c)
  | _ -> assert false

let is_integer_dtype = function
  | D.I32 | D.I64 | D.Uint32 | D.I8 | D.I16 | D.I2 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint2 | D.Uint4 | D.Uint64 ->
      true
  | D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
  | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
  | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Float0 | D.Bool | D.Complex64
  | D.Complex128 ->
      false

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let reject_complex name v =
  if is_complex_dtype (dtype v) then
    raise
      (Errors.jax_type_error
         (name ^ " does not support complex-valued inputs."))

let real_only name x =
  let x = promote1 x in
  reject_complex name x;
  x

let real_only2 name a b =
  let a, b = promote2 a b in
  reject_complex name a;
  (a, b)

let real_only3 name a b c =
  let a, b, c = promote3 a b c in
  reject_complex name a;
  (a, b, c)

let single_or_double x =
  match dtype x with
  | D.F32 | D.F64 -> ()
  | d ->
      raise
        (Errors.jax_type_error
           (Printf.sprintf
              "x.dtype=%s is not supported, see docstring for supported types."
              (D.short_name d)))

let eps_of v =
  match dtype v with
  | D.F64 | D.Complex128 -> epsilon_float
  | D.F16 -> 0.0009765625
  | D.Bf16 -> 0.0078125
  | d when Option.is_some (Float_format.of_dtype d) -> (Dtypes.finfo d).eps
  | _ -> 1.1920928955078125e-07

let eps50 v = scalar_of v (eps_of v *. 50.0)

let select_default default condlist choicelist =
  NL.select ~default condlist choicelist

let vectorize_scalar f args =
  let shapes = List.map shape args in
  let broadcast = NL.broadcast_shapes_n shapes in
  let rank = Array.length broadcast in
  let filled =
    List.map
      (fun s -> Array.append (Array.make (rank - Array.length s) 1) s)
      shapes
  in
  let squeezed =
    List.map
      (fun a ->
        let s = shape a in
        let axes =
          List.filter (fun i -> s.(i) = 1) (List.init (Array.length s) Fun.id)
        in
        if axes = [] then a else NL.squeeze ~axis:(Array.of_list axes) a)
      args
  in
  let func = ref f and expand = ref [] in
  for pos = rank - 1 downto 0 do
    let in_axes =
      List.map (fun s -> if s.(pos) = 1 then None else Some 0) filled
    in
    if List.for_all Option.is_none in_axes then expand := pos :: !expand
    else begin
      let inner = !func in
      func := fun xs -> Interpreters.Batching.vmap inner in_axes xs
    end
  done;
  let out = !func squeezed in
  match !expand with
  | [] -> out
  | dims -> List.map (fun v -> NL.expand_dims v (Array.of_list dims)) out

type piece = Piece_const of T.value | Piece_fn of (T.value list -> T.value)

let piecewise_body dt x condlist funclist extra =
  let lead =
    let first =
      if (shape condlist).(0) = 1 then condlist
      else
        let sh = shape condlist in
        bind1
          (T.Slice
             {
               start_indices = Array.make (Array.length sh) 0;
               limit_indices = Array.mapi (fun d e -> if d = 0 then 1 else e) sh;
               strides = None;
             })
          [ condlist ]
    in
    AC.zeros_like first
  in
  let indices =
    NL.argmax ~axis:0
      (RED.cumsum ~axis:0 (NL.concatenate ~axis:0 [ lead; condlist ]))
  in
  let branches =
    List.map
      (fun piece ops ->
        match (piece, ops) with
        | Piece_fn f, [ v ] ->
            [ C.convert_like ~weak:false (f (v :: extra)) dt ]
        | Piece_const c, [ _ ] -> [ C.convert_like ~weak:false (NL.copy c) dt ]
        | _ -> invalid_arg "piecewise: arity")
      funclist
  in
  let switch = function
    | [ i; v ] -> L.switch i branches [ v ]
    | _ -> invalid_arg "piecewise: arity"
  in
  List.hd (vectorize_scalar switch [ indices; x ])

let piecewise ~key ?(extra = []) x conds funcs =
  let n = List.length funcs in
  let rotated =
    if n = List.length conds + 1 then
      List.nth funcs (n - 1) :: List.filteri (fun i _ -> i < n - 1) funcs
    else if n = List.length conds then Piece_const (py_int 0) :: funcs
    else invalid_arg "piecewise: function count"
  in
  let expand c =
    let sh = shape c in
    let c = C.convert_like ~weak:false c D.Bool in
    bind1
      (T.Broadcast_in_dim
         {
           shape = Array.append [| 1 |] sh;
           dims = Array.init (Array.length sh) (fun d -> d + 1);
         })
      [ c ]
  in
  let condlist =
    match List.map expand conds with
    | [ only ] -> only
    | parts -> bind1 (T.Concatenate 0) parts
  in
  let const_vals =
    List.filter_map (function Piece_const c -> Some c | _ -> None) rotated
  in
  let n_const = List.length const_vals in
  let n_extra = List.length extra in
  JU.jit1 "_piecewise" ~key
    (function
      | x :: condlist :: rest ->
          let consts = List.filteri (fun i _ -> i < n_const) rest in
          let extra = List.filteri (fun i _ -> i >= n_const) rest in
          if List.length extra <> n_extra then invalid_arg "piecewise: arity";
          let remaining = ref consts in
          let funclist =
            List.map
              (function
                | Piece_const _ ->
                    let c = List.hd !remaining in
                    remaining := List.tl !remaining;
                    Piece_const c
                | piece -> piece)
              rotated
          in
          piecewise_body (dtype x) x condlist funclist extra
      | _ -> invalid_arg "piecewise: arity")
    ((x :: condlist :: const_vals) @ extra)

let constf v = Piece_const (py_float v)
let const_val v = Piece_const v
let piece_fn f = Piece_fn (function [ x ] -> f x | _ -> assert false)
let fresh_key () = string_of_int (C.fresh_id ())
let gammaln x = bind1 T.Lgamma [ real_only "gammaln" x ]

let gammasgn x =
  let x = real_only "gammasgn" x in
  let floor_x = L.floor x in
  let x_negative = lt_ x 0.0 in
  let c1 = bor (band x_negative (UF.equal x floor_x)) (UF.isnan x) in
  let c2 =
    bor
      (band x_negative (ne_ (UF.remainder floor_x (py_int 2)) 0.0))
      (band (eq_ x 0.0) (UF.signbit x))
  in
  select_default (lax_const x 1.0) [ c1; c2 ]
    [ lax_const x Float_format.nan; lax_const x (-1.0) ]

let lanczos_g = 7.0

let complex_loggamma z =
  let needs_reflection = lt_ (UF.real z) 0.5 in
  let z_lanczos = where_ needs_reflection (scalar_of z 1.0 -! z) z in
  let zz = sub_lit z_lanczos 1.0 in
  let n = ndim zz in
  let coeffs = c_lanczos_coeffs in
  let ks =
    NL.expand_dims
      (T.Concrete
         (Nd.of_floats (dtype z) [| 8 |]
            (Array.init 8 (fun i -> float_of_int (i + 1)))))
      (Array.init n Fun.id)
  in
  let tail =
    NL.expand_dims (coeff_of z (Array.sub coeffs 1 8)) (Array.init n Fun.id)
  in
  let ag =
    scalar_of z coeffs.(0)
    +! RED.sum ~axis:[| -1 |] (tail /! (NL.expand_dims zz [| n |] +! ks))
  in
  let t = add_lit (add_lit zz lanczos_g) 0.5 in
  let half_log2pi = scalar_of z (0.5 *. Float.log (2.0 *. Float.pi)) in
  let log_gamma_lanczos =
    half_log2pi +! (add_lit zz 0.5 *! UF.log t) -! t +! UF.log ag
  in
  let z_safe = where_ needs_reflection z (AC.full_like z 0.5) in
  let reflected =
    UF.log (scalar_of z Float.pi)
    -! UF.log (UF.sin (scalar_of z Float.pi *! z_safe))
    -! log_gamma_lanczos
  in
  where_ needs_reflection reflected log_gamma_lanczos

let complex_of_parts z re im = NL.astype (bind1 T.Complex [ re; im ]) (dtype z)

let complex_nan z =
  let real_dt = if dtype z = D.Complex128 then D.F64 else D.F32 in
  let nan = const_full real_dt [||] Float_format.nan in
  complex_of_parts z nan nan

let complex_pole_or_nan z =
  let re = UF.real z in
  let is_nan = UF.isnan z in
  let is_pole =
    band (band (eq_ (UF.imag z) 0.0) (UF.equal re (UF.floor re))) (le_ re 0.0)
  in
  bor is_pole is_nan

let complex_gamma z =
  let safe = complex_pole_or_nan z in
  let z_safe = where_ safe (AC.ones_like z) z in
  where_ safe (complex_nan z) (UF.exp (complex_loggamma z_safe))

let complex_loggamma_scipy z =
  let safe = complex_pole_or_nan z in
  let z_safe = where_ safe (AC.ones_like z) z in
  let idt = default_int () in
  let n =
    UF.maximum (py_int 0)
      (NL.astype (UF.ceil (lit_sub 0.5 (UF.real z_safe))) idt)
  in
  let result =
    L.while_loop
      (function
        | [ k; _ ] -> RED.any (UF.less k n) | _ -> failwith "loggamma: state")
      (function
        | [ k; sum_log ] ->
            let mask = UF.less k n in
            let zk = where_ mask (z_safe +! k) (AC.ones_like z_safe) in
            [
              k +! py_int 1;
              sum_log +! where_ mask (UF.log zk) (AC.zeros_like sum_log);
            ]
        | _ -> failwith "loggamma: state")
      [ py_int 0; AC.zeros_like z_safe ]
  in
  let sum_log = List.nth result 1 in
  let res = complex_loggamma (z_safe +! n) -! sum_log in
  where_ safe (complex_nan z) res

let complex_loggamma_scipy a0 =
  Eager_jit.one ~jit:"_complex_loggamma_scipy"
    "scipy.special._complex_loggamma_scipy" () [] [ a0 ] (fun _ vals ->
      match vals with [ a0 ] -> complex_loggamma_scipy a0 | _ -> assert false)

let loggamma x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then complex_loggamma_scipy x
  else
    let res = bind1 T.Lgamma [ x ] in
    where_py (gt_ x 0.0) res Float_format.nan

let gammasgn_jit a0 =
  Eager_jit.one ~jit:"gammasgn" "scipy.special.gammasgn" () [] [ a0 ]
    (fun _ vals -> match vals with [ a0 ] -> gammasgn a0 | _ -> assert false)

let gamma x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then complex_gamma x
  else gammasgn_jit x *! L.exp (bind1 T.Lgamma [ x ])

let algdiv a b =
  let c0 = 0.0833333333333333
  and c1 = -0.00277777777760991
  and c2 = 0.00079365066682539
  and c3 = -0.00059520293135187
  and c4 = 0.000837308034031215
  and c5 = -0.00165322962780713 in
  let h = a /! b in
  let c = h /! lit_add 1.0 h in
  let x = h /! lit_add 1.0 h in
  let d = b +! sub_lit a 0.5 in
  let x2 = x *! x in
  let s3 = lit_add 1.0 (x +! x2) in
  let s5 = lit_add 1.0 (x +! (x2 *! s3)) in
  let s7 = lit_add 1.0 (x +! (x2 *! s5)) in
  let s9 = lit_add 1.0 (x +! (x2 *! s7)) in
  let s11 = lit_add 1.0 (x +! (x2 *! s9)) in
  let t = UF.power (lit_div 1.0 b) (py_int 2) in
  let w = (lit_mul c5 s11 *! t) +! lit_mul c4 s9 in
  let w = (w *! t) +! lit_mul c3 s7 in
  let w = (w *! t) +! lit_mul c2 s5 in
  let w = (w *! t) +! lit_mul c1 s3 in
  let w = add_lit (w *! t) c0 in
  let w = w *! (c /! b) in
  let u = d *! L.log1p (a /! b) in
  let v = a *! sub_lit (L.log b) 1.0 in
  where_ (UF.less_equal u v) (w -! v -! u) (w -! u -! v)

let lgamma_prim v = bind1 T.Lgamma [ v ]

let betaln a b =
  let a, b = promote2 a b in
  let a, b = (UF.minimum a b, UF.maximum a b) in
  let small_b = lgamma_prim a +! (lgamma_prim b -! lgamma_prim (a +! b)) in
  let large_b = lgamma_prim a +! algdiv a b in
  where_ (lt_ b 8.0) small_b large_b

let factorial ?(exact = false) n =
  if exact then failwith "factorial with exact=True";
  let n = promote1 n in
  where_ (lt_ n 0.0) (py_int 0) (L.exp (lgamma_prim (add_lit n 1.0)))

let rec comb ?(repetition = false) n_ k_ =
  let n, k = promote2 n_ k_ in
  if repetition then
    let cond = band (eq_ k 0.0) (ge_ n 0.0) in
    let result = comb ~repetition:false (sub_lit (n +! k) 1.0) k in
    where_px cond 1.0 result
  else
    let cond = band (band (UF.less_equal k n) (ge_ n 0.0)) (ge_ k 0.0) in
    let safe_n = where_py cond n 0.0 in
    let safe_k = where_py cond k 0.0 in
    let result =
      L.exp
        (gammaln (add_lit safe_n 1.0)
        -! gammaln (add_lit safe_k 1.0)
        -! gammaln (add_lit safe_n 1.0 -! safe_k))
    in
    where_py cond result 0.0

let beta a b =
  let a, b = real_only2 "beta" a b in
  let sign = gammasgn_jit a *! gammasgn_jit b *! gammasgn_jit (a +! b) in
  sign *! L.exp (betaln a b)

let betainc a b x =
  let a, b, x = real_only3 "betainc" a b x in
  bind1 T.Regularized_incomplete_beta [ a; b; x ]

let digamma x = bind1 T.Digamma [ real_only "digamma" x ]

let gammainc a x =
  let a, x = promote2 a x in
  bind1 T.Igamma [ a; x ]

let gammaincc a x =
  let a, x = promote2 a x in
  bind1 T.Igammac [ a; x ]

let lax_igamma a x = bind1 T.Igamma [ a; x ]
let erf x = bind1 T.Erf [ promote1 x ]
let erfc x = bind1 T.Erfc [ promote1 x ]
let erfinv x = bind1 T.Erf_inv [ real_only "erfinv" x ]
let expit x = bind1 T.Logistic [ promote1 x ]

let rec logit x =
  List.hd
    (CD.custom_jvp ~name:"logit"
       (function
         | [ x ] ->
             let x = promote1 x in
             [ L.log (L.div x (L.sub (scalar_of x 1.0) x)) ]
         | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ g ] ->
             ([ logit x ], [ L.div g (L.mul x (L.sub (scalar_of x 1.0) x)) ])
         | _ -> assert false)
       [ x ])

let xlogy_primal = function
  | [ x; y ] ->
      let x, y = promote2 x y in
      let x_ok = ne_ x 0.0 in
      [ where_ (bor x_ok (UF.isnan y)) (L.mul x (L.log y)) (AC.zeros_like x) ]
  | _ -> assert false

let rec xlogy x y =
  List.hd
    (CD.custom_jvp ~name:"xlogy" ~jvp_name:"_xlogy_jvp" xlogy_primal
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x; y ], [ x_dot; y_dot ] ->
             let result = xlogy x y in
             let out = (x_dot *! L.log y) +! (y_dot *! x /! y) in
             ([ result ], [ NL.astype out (dtype result) ])
         | _ -> assert false)
       [ x; y ])

let xlog1py_primal = function
  | [ x; y ] ->
      let x, y = promote2 x y in
      let x_ok = ne_ x 0.0 in
      [ where_ (bor x_ok (UF.isnan y)) (L.mul x (L.log1p y)) (AC.zeros_like x) ]
  | _ -> assert false

let rec xlog1py x y =
  List.hd
    (CD.custom_jvp ~name:"xlog1py" ~jvp_name:"_xlog1py_jvp" xlog1py_primal
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x; y ], [ x_dot; y_dot ] ->
             let result = xlog1py x y in
             let out = (x_dot *! L.log1p y) +! (y_dot *! x /! lit_add 1.0 y) in
             ([ result ], [ NL.astype out (dtype result) ])
         | _ -> assert false)
       [ x; y ])

let rec xlogx x =
  List.hd
    (CD.custom_jvp ~name:"_xlogx" ~jvp_name:"_xlogx_jvp"
       (function [ x ] -> [ xlogy x x ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ x_dot ] -> ([ xlogx x ], [ x_dot *! add_lit (L.log x) 1.0 ])
         | _ -> assert false)
       [ x ])

let entr x =
  let x = real_only "entr" x in
  L.select
    (L.lt x (scalar_of x 0.0))
    (L.full_like x Float.neg_infinity)
    (L.neg (xlogx x))

let boxcox x lmbda =
  let x, lmbda = promote2 x lmbda in
  let is_zero = eq_ lmbda 0.0 in
  let safe = where_ is_zero (AC.ones_like lmbda) lmbda in
  let log_x = UF.log x in
  let power = UF.expm1 (safe *! log_x) /! safe in
  where_ is_zero log_x power

let boxcox1p x lmbda =
  let x, lmbda = promote2 x lmbda in
  let is_zero = eq_ lmbda 0.0 in
  let safe = where_ is_zero (AC.ones_like lmbda) lmbda in
  let log1p_x = UF.log1p x in
  let power = UF.expm1 (safe *! log1p_x) /! safe in
  where_ is_zero log1p_x power

let multigammaln a d =
  let a = promote1 a in
  let d_ = scalar_of a (float_of_int d) in
  let constant =
    L.mul
      (L.mul (L.mul (scalar_of a 0.25) d_) (L.sub d_ (scalar_of a 1.0)))
      (L.log (scalar_of a Float.pi))
  in
  let b =
    L.div (NL.arange ~dtype:(dtype a) (float_of_int d)) (scalar_of a 2.0)
  in
  let rank = ndim a in
  let res =
    RED.sum ~axis:[| -1 |]
      (gammaln
         (NL.expand_dims a [| rank |]
         -! NL.expand_dims b (Array.init rank Fun.id)))
  in
  res +! constant

let rel_entr p q =
  let p, q = real_only2 "rel_entr" p q in
  let zero = lax_const p 0.0 in
  let both = L.bitwise_and (L.gt p zero) (L.gt q zero) in
  let one_zero = L.bitwise_and (L.eq p zero) (L.ge q zero) in
  let safe_p = where_ both p (py_int 1) in
  let safe_q = where_ both q (py_int 1) in
  let log_val = L.sub (xlogx safe_p) (xlogy safe_p safe_q) in
  where_ both log_val (where_py one_zero zero Float.infinity)

let kl_div p q =
  let p, q = real_only2 "kl_div" p q in
  rel_entr p q -! p +! q

let i0e x = bind1 T.Bessel_i0e [ promote1 x ]

let i0 x =
  let x = promote1 x in
  L.mul (L.exp (L.abs x)) (bind1 T.Bessel_i0e [ x ])

let i1e x = bind1 T.Bessel_i1e [ promote1 x ]

let i1 x =
  let x = promote1 x in
  L.mul (L.exp (L.abs x)) (bind1 T.Bessel_i1e [ x ])

let wofz_upper z =
  let re = L.real z in
  let cplx a b = L.complex (scalar_of re a) (scalar_of re b) in
  let l = cplx 4.756828460010884 0.0 in
  let iz = L.complex (L.neg (L.imag z)) re in
  let denom = l -! iz in
  let zz = (l +! iz) /! denom in
  let p = polyval (coeff_of zz c_wofz_c) zz in
  let one_over_sqrtpi = cplx (1.0 /. sqrt Float.pi) 0.0 in
  (lit_mul 2.0 p /! (denom *! denom)) +! (one_over_sqrtpi /! denom)

let wofz_primal z =
  let sign = L.ge (L.imag z) (scalar_of (L.real z) 0.0) in
  let z_upper = L.select sign z (L.neg z) in
  let w_upper = wofz_upper z_upper in
  let correction = lit_mul 2.0 (L.exp (L.neg (z *! z))) -! w_upper in
  L.select sign w_upper correction

let rec wofz_custom z =
  List.hd
    (CD.custom_jvp ~name:"_wofz"
       (function [ z ] -> [ wofz_primal z ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ z ], [ g ] ->
             let ans = wofz_custom z in
             let re = L.real z in
             ( [ ans ],
               [
                 g
                 *! (L.neg (lit_mul 2.0 z *! ans)
                    +! L.complex (scalar_of re 0.0)
                         (scalar_of re (2.0 /. sqrt Float.pi)));
               ] )
         | _ -> assert false)
       [ z ])

let wofz z =
  let dt =
    match dtype z with
    | D.F64 | D.Complex128 | D.I64 | D.Uint64 -> D.Complex128
    | _ -> D.Complex64
  in
  wofz_custom (if dtype z = dt then z else NL.astype z dt)

let erfcx_asymptotic x nterms =
  let coeffs =
    [|
      7918.06640625;
      -1055.7421875;
      162.421875;
      -29.53125;
      6.5625;
      -1.875;
      0.75;
      -0.5;
      1.0;
    |]
  in
  let t = scalar_of x 1.0 /! L.square x in
  let p =
    polyval
      (coeff_of x (Array.sub coeffs (Array.length coeffs - nterms) nterms))
      t
  in
  p /! (x *! scalar_of x (sqrt Float.pi))

let erfcx_impl x nterms =
  let threshold =
    match dtype x with
    | D.F64 -> sqrt (log Float.max_float)
    | _ -> q32 (sqrt (q32 (log 3.4028234663852886e38)))
  in
  let large = gt_ x threshold in
  let safe_x = L.select large (L.full_like x 1.0) x in
  let direct = L.exp (L.square safe_x) *! bind1 T.Erfc [ safe_x ] in
  let asymp = erfcx_asymptotic x nterms in
  L.select large asymp direct

let erfcx_real x =
  match dtype x with
  | D.F64 -> erfcx_impl x 9
  | D.F32 -> erfcx_impl x 5
  | d -> NL.astype (erfcx_impl (NL.astype x D.F32) 5) d

let rec erfcx_custom x =
  List.hd
    (CD.custom_jvp ~name:"_erfcx"
       (function [ x ] -> [ erfcx_real x ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ g ] ->
             let ans = erfcx_custom x in
             ( [ ans ],
               [
                 g
                 *! ((lit_mul 2.0 x *! ans)
                    -! scalar_of x (2.0 /. sqrt Float.pi));
               ] )
         | _ -> assert false)
       [ x ])

let erfcx x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then
    let iz = L.complex (L.neg (L.imag x)) (L.real x) in
    wofz_custom iz
  else erfcx_custom x

let dawsn_impl x =
  let sign = UF.sign x in
  let ax = L.abs x in
  let an = coeff_of x c_dawsn_an
  and ad = coeff_of x c_dawsn_ad
  and bn = coeff_of x c_dawsn_bn
  and bd = coeff_of x c_dawsn_bd
  and cn = coeff_of x c_dawsn_cn
  and cd = coeff_of x c_dawsn_cd in
  let c v = scalar_of x v in
  let ax2 = L.square ax in
  let t = c 1.0 /! ax2 in
  let safe_ax_r1 = L.select (UF.less ax (c 3.25)) ax (L.full_like ax 1.0) in
  let safe_ax2_r1 = L.square safe_ax_r1 in
  let val_r1 = safe_ax_r1 *! polyval an safe_ax2_r1 /! polyval ad safe_ax2_r1 in
  let m2 () = band (UF.greater_equal ax (c 3.25)) (UF.less ax (c 6.25)) in
  let safe_t_r2 = L.select (m2 ()) t (L.full_like t 1.0) in
  let safe_ax_r2 = L.select (m2 ()) ax (L.full_like ax 1.0) in
  let val_r2 =
    c 0.5 /! safe_ax_r2
    *! (c 1.0 +! (safe_t_r2 *! polyval bn safe_t_r2 /! polyval bd safe_t_r2))
  in
  let m3 () = UF.greater_equal ax (c 6.25) in
  let safe_t_r3 = L.select (m3 ()) t (L.full_like t 1.0) in
  let safe_ax_r3 = L.select (m3 ()) ax (L.full_like ax 1.0) in
  let val_r3 =
    c 0.5 /! safe_ax_r3
    *! (c 1.0 +! (safe_t_r3 *! polyval cn safe_t_r3 /! polyval cd safe_t_r3))
  in
  let result =
    L.select
      (UF.less ax (c 3.25))
      val_r1
      (L.select (UF.less ax (c 6.25)) val_r2 val_r3)
  in
  sign *! result

let dawsn_real x =
  match dtype x with
  | D.F32 | D.F64 -> dawsn_impl x
  | d -> NL.astype (dawsn_impl (NL.astype x D.F32)) d

let rec dawsn_custom x =
  List.hd
    (CD.custom_jvp ~name:"_dawsn"
       (function [ x ] -> [ dawsn_real x ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ g ] ->
             let ans = dawsn_custom x in
             ( [ ans ],
               [ g *! (scalar_of x 1.0 -! (scalar_of x 2.0 *! x *! ans)) ] )
         | _ -> assert false)
       [ x ])

let dawsn x = dawsn_custom (real_only "dawsn" x)

let half_sqrt_2_of x =
  match dtype x with D.F32 -> 0.5 *. q32 (sqrt 2.0) | _ -> 0.5 *. sqrt 2.0

let ndtr_core x =
  let half_sqrt_2 = half_sqrt_2_of x in
  let w = mul_lit x half_sqrt_2 in
  let z = L.abs w in
  let y =
    L.select
      (L.lt z (scalar_of x half_sqrt_2))
      (lit_add 1.0 (bind1 T.Erf [ w ]))
      (L.select
         (L.gt w (scalar_of x 0.0))
         (lit_sub 2.0 (bind1 T.Erfc [ z ]))
         (bind1 T.Erfc [ z ]))
  in
  lit_mul 0.5 y

let ndtr x =
  single_or_double x;
  ndtr_core x

let ndtri p =
  single_or_double p;
  let sh = shape p in
  let p0 = coeff_of p c_ndtri_p0
  and q0 = coeff_of p c_ndtri_q0
  and p1 = coeff_of p c_ndtri_p1
  and q1 = coeff_of p c_ndtri_q1
  and p2 = coeff_of p c_ndtri_p2
  and q2 = coeff_of p c_ndtri_q2 in
  let c v = lax_const p v in
  let maybe = where_ (UF.greater p (c (-.Float.expm1 (-2.0)))) (c 1.0 -! p) p in
  let sanitized =
    where_ (UF.equal maybe (c 0.0)) (L.broadcast (c 0.5) sh) maybe
  in
  let w = sanitized -! c 0.5 in
  let ww = L.square w in
  let x_big = w +! (w *! ww *! (polyval p0 ww /! polyval q0 ww)) in
  let x_big = x_big *! c (-.sqrt (2.0 *. Float.pi)) in
  let z = L.sqrt (c (-2.0) *! L.log sanitized) in
  let first = z -! (L.log z /! z) in
  let inv () = lit_div 1.0 z in
  let num2 = polyval p2 (inv ()) in
  let den2 = polyval q2 (inv ()) in
  let second_small = num2 /! den2 /! z in
  let num1 = polyval p1 (inv ()) in
  let den1 = polyval q1 (inv ()) in
  let second_other = num1 /! den1 /! z in
  let x_small = first -! second_small in
  let x_other = first -! second_other in
  let x =
    where_
      (UF.greater sanitized (c (exp (-2.0))))
      x_big
      (where_ (UF.greater_equal z (c 8.0)) x_small x_other)
  in
  let x = where_ (UF.greater p (c (1.0 -. exp (-2.0)))) x (neg_ x) in
  let infinity = L.broadcast (c Float.infinity) sh in
  where_
    (UF.equal p (c 0.0))
    (neg_ infinity)
    (where_ (UF.equal p (c 1.0)) infinity x)

let double_factorial m =
  let rec loop k acc =
    if k <= 1 then acc else loop (k - 2) (acc *. float_of_int k)
  in
  loop m 1.0

let log_ndtr_asymptotic_series x series_order =
  if series_order <= 0 then scalar_of x 1.0
  else begin
    let x2 = L.square x in
    let even = ref (AC.zeros_like x) in
    let odd = ref (AC.zeros_like x) in
    let x2n = ref x2 in
    for n = 1 to series_order do
      let y = scalar_of x (double_factorial ((2 * n) - 1)) /! !x2n in
      if n mod 2 = 1 then odd := !odd +! y else even := !even +! y;
      x2n := !x2n *! x2
    done;
    lit_add 1.0 !even -! !odd
  end

let log_ndtr_lower x series_order =
  let x2 = L.square x in
  let log_scale =
    (scalar_of x (-0.5) *! x2)
    -! L.log (neg_ x)
    -! scalar_of x (0.5 *. log (2.0 *. Float.pi))
  in
  log_scale +! L.log (log_ndtr_asymptotic_series x series_order)

let log_ndtr_primal series_order x =
  single_or_double x;
  let lower, upper =
    match dtype x with D.F64 -> (-20.0, 8.0) | _ -> (-10.0, 5.0)
  in
  let lower_v = scalar_of x lower in
  let gt_upper = L.gt x (scalar_of x upper) in
  let ndtr_arg = where_ gt_upper (neg_ x) (L.max x lower_v) in
  let ndtr_res = ndtr_core ndtr_arg in
  where_ gt_upper (neg_ ndtr_res)
    (where_ (L.gt x lower_v) (L.log ndtr_res)
       (log_ndtr_lower (L.min x lower_v) series_order))

let norm_logpdf_constant = log (sqrt (2.0 *. Float.pi))

let norm_logpdf x =
  L.sub
    (L.mul (scalar_of x (-0.5)) (L.square x))
    (scalar_of x norm_logpdf_constant)

let rec log_ndtr ?(series_order = 3) x =
  if series_order < 0 then invalid_arg "series_order must be non-negative.";
  if series_order > 30 then invalid_arg "series_order must be <= 30.";
  single_or_double x;
  List.hd
    (CD.custom_jvp ~name:"log_ndtr" ~jvp_name:"_log_ndtr_jvp"
       (function
         | [ x ] -> [ log_ndtr_primal series_order x ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ t ] ->
             let ans = log_ndtr ~series_order x in
             ([ ans ], [ L.mul t (L.exp (L.sub (norm_logpdf x) ans)) ])
         | _ -> assert false)
       [ x ])

let polygamma n x =
  if not (is_integer_dtype (dtype n)) then
    invalid_arg "Argument `n` to polygamma must be of integer type.";
  let n, x = real_only2 "polygamma" n x in
  bind1 T.Polygamma [ n; x ]

let zeta_series x q =
  let s, a = promote2 x q in
  let is_f32 = dtype a = D.F32 in
  let count = if is_f32 then 8 else 16 in
  let lit v = scalar_of a v in
  let big_n = lit (float_of_int count) in
  let s_ = NL.expand_dims s [| -1 |] and a_ = NL.expand_dims a [| -1 |] in
  let k =
    NL.expand_dims
      (T.Concrete
         (Nd.of_floats (dtype a) [| count |] (Array.init count float_of_int)))
      (Array.init (ndim a) Fun.id)
  in
  let series = RED.sum ~axis:[| -1 |] (UF.power (a_ +! k) (neg_ s_)) in
  let shifted () = a +! big_n in
  let integral = L.div (UF.power (shifted ()) (lit 1.0 -! s)) (s -! lit 1.0) in
  let t0 = UF.power (shifted ()) (neg_ s) in
  let m =
    NL.expand_dims
      (T.Concrete
         (Nd.of_floats (dtype a)
            [| 2 * count |]
            (Array.init (2 * count) float_of_int)))
      (Array.init (ndim s) Fun.id)
  in
  let s_over_a = (s_ +! m) /! (a_ +! big_n) in
  let t1 = RED.cumprod ~axis:(-1) s_over_a in
  let rank = ndim t1 in
  let begins = Array.make rank 0 and ends = shape t1 in
  let strides = Array.make rank 1 in
  strides.(rank - 1) <- 2;
  let t1 = L.slice ~strides t1 begins ends in
  let fmax = if is_f32 then 3.4028234663852886e38 else Float.max_float in
  let t1 = NL.clip_strong_max (lax_const a fmax) t1 in
  let terms = (shape t1).(rank - 1) in
  let coefs =
    T.Concrete
      (Nd.of_floats (dtype a)
         (Array.append (Array.make (ndim a) 1) [| terms |])
         (Array.sub c_bernoulli_coefs 0 terms))
  in
  let t1 = t1 /! coefs in
  let total = t0 *! (lit 0.5 +! RED.sum ~axis:[| -1 |] t1) in
  series +! integral +! total

let zeta ?q x =
  match q with
  | None ->
      failwith
        "Riemann zeta function not implemented; pass q != None to compute the \
         Hurwitz Zeta function."
  | Some q ->
      let primal = function
        | [ x; q ] ->
            let x, q = promote2 x q in
            [ bind1 T.Zeta [ x; q ] ]
        | _ -> failwith "zeta: arity"
      in
      List.hd
        (CD.custom_jvp ~name:"zeta" primal
           ~jvp:(fun primals tangents ->
             Interpreters.Ad.jvp
               (function
                 | [ x; q ] -> [ zeta_series x q ] | _ -> failwith "zeta: arity")
               primals tangents)
           [ x; q ])

let reject_bool x =
  if dtype x = D.Bool then
    raise (Errors.jax_type_error "sub does not accept dtype bool.")

let softmax ?axis x =
  reject_bool x;
  match axis with
  | Some ax -> NN.softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let log_softmax ?axis x =
  reject_bool x;
  match axis with
  | Some ax -> NN.log_softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.log_softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let poch_primal z m =
  let z, m = real_only2 "poch" z m in
  where_ (eq_ m 0.0) (lax_const z 1.0) (gamma (z +! m) /! gamma z)

let rec poch z m =
  List.hd
    (CD.custom_jvp ~name:"poch"
       (function [ z; m ] -> [ poch_primal z m ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ z; m ], [ dz; dm ] ->
             let out = poch z m in
             let wrt_z = (digamma (z +! m) -! digamma z) *! poch z m *! dz in
             let wrt_m = digamma (z +! m) *! poch z m *! dm in
             ([ out ], [ Ad_util.add_any wrt_z wrt_m ])
         | _ -> assert false)
       [ z; m ])

let spence_poly w =
  let a = coeff_of w c_spence_poly_a and b = coeff_of w c_spence_poly_b in
  neg_ w *! polyval a w /! polyval b w

let spence_calc x =
  let x2_bool = gt_ x 2.0 in
  let x =
    piecewise ~key:(fresh_key ()) x [ x2_bool ]
      [ piece_fn (fun x -> lit_div 1.0 x); piece_fn (fun x -> x) ]
  in
  let x1_5_bool = gt_ x 1.5 in
  let x_5_bool = lt_ x 0.5 in
  let x2_bool = bor x2_bool x1_5_bool in
  let w =
    piecewise ~key:(fresh_key ()) x [ x1_5_bool; x_5_bool ]
      [
        piece_fn (fun x -> sub_lit (lit_div 1.0 x) 1.0);
        piece_fn (fun x -> neg_ x);
        piece_fn (fun x -> sub_lit x 1.0);
      ]
  in
  let y = spence_poly w in
  let pi2_6 = (Float.pi ** 2.0) /. 6.0 in
  let y_flag_one =
    scalar_of x pi2_6 -! (UF.log x *! UF.log (lit_sub 1.0 x)) -! y
  in
  let y = where_ x_5_bool y_flag_one y in
  let y_flag_two =
    (scalar_of x (-0.5) *! UF.power (UF.log x) (py_int 2)) -! y
  in
  where_ x2_bool y_flag_two y

let spence_body x =
  piecewise ~key:"spence" x
    [ lt_ x 0.0; eq_ x 1.0; eq_ x 0.0 ]
    [
      constf Float_format.nan;
      Piece_const (py_int 0);
      constf ((Float.pi ** 2.0) /. 6.0);
      piece_fn spence_calc;
    ]

let spence x =
  (match dtype x with
  | D.F32 | D.F64 -> ()
  | d ->
      raise
        (Errors.jax_type_error
           (Printf.sprintf
              "x.dtype=%s is not supported, see docstring for supported types."
              (D.short_name d))));
  spence_body x

let euler_gamma = 0.5772156649015329

let sici_series x =
  let t () = x *! x in
  let t1 = t () in
  let si_v =
    x
    *! polyval (coeff_of x c_sici_series_sn) t1
    /! polyval (coeff_of x c_sici_series_sd) t1
  in
  let si = where_px (eq_ x 0.0) 0.0 si_v in
  let t2 = t () in
  let ci_v =
    scalar_of x euler_gamma +! UF.log x
    +! t2
       *! polyval (coeff_of x c_sici_series_cn) t2
       /! polyval (coeff_of x c_sici_series_cd) t2
  in
  let ci = where_px (eq_ x 0.0) Float.neg_infinity ci_v in
  (si, ci)

let sici_asympt x =
  let s = UF.sin x and c = UF.cos x in
  let z = lit_div 1.0 (x *! x) in
  let p cs = polyval (coeff_of x cs) z in
  let f4 = p c_sici_asympt_fn4 /! (x *! p c_sici_asympt_fd4) in
  let g4 = z *! p c_sici_asympt_gn4 /! p c_sici_asympt_gd4 in
  let f8 = p c_sici_asympt_fn8 /! (x *! p c_sici_asympt_fd8) in
  let g8 = z *! p c_sici_asympt_gn8 /! p c_sici_asympt_gd8 in
  let mask = lt_ x 8.0 in
  let f = where_ mask f4 f8 in
  let g = where_ mask g4 g8 in
  let si = lit_sub (Float.pi /. 2.0) (f *! c) -! (g *! s) in
  let ci = (f *! s) -! (g *! c) in
  (si, ci)

let sici_approx x =
  let si = lit_sub (Float.pi /. 2.0) (UF.cos x /! x) in
  let ci = UF.sin x /! x in
  let si = where_px (UF.isposinf x) (Float.pi /. 2.0) si in
  let ci = where_px (UF.isposinf x) 0.0 ci in
  (si, ci)

let sici_primal x =
  let x = real_only "sici" x in
  let x_abs = UF.abs x in
  let si_series, ci_series = sici_series x_abs in
  let si_asymp, ci_asymp = sici_asympt x_abs in
  let si_approx, ci_approx = sici_approx x_abs in
  let cond1 = le_ x_abs 4.0 in
  let cond2 = band (gt_ x_abs 4.0) (le_ x_abs 1e9) in
  let si = select_default si_approx [ cond1; cond2 ] [ si_series; si_asymp ] in
  let ci = select_default ci_approx [ cond1; cond2 ] [ ci_series; ci_asymp ] in
  let si = UF.sign x *! si in
  let ci = where_px (UF.isneginf x) Float_format.nan ci in
  [ si; ci ]

let sici_jit a0 =
  Eager_jit.many ~jit:"sici" "scipy.special.sici" () [] [ a0 ] (fun _ vals ->
      match vals with [ a0 ] -> sici_primal a0 | _ -> assert false)

let sici x =
  CD.custom_jvp ~name:"sici" ~jvp_name:"sici_jvp"
    (function [ x ] -> sici_jit x | _ -> assert false)
    ~jvp:(fun primals tangents ->
      match (primals, tangents) with
      | [ p ], [ t ] -> (
          match
            JU.jit "sici_jvp"
              (function
                | [ p; t ] ->
                    let out = sici_jit p in
                    let sin_term = UF.sinc (div_lit p Float.pi) in
                    let cos_term = UF.cos p /! p in
                    out @ [ sin_term *! t; cos_term *! t ]
                | _ -> assert false)
              [ p; t ]
          with
          | [ si; ci; ts; tc ] -> ([ si; ci ], [ ts; tc ])
          | _ -> assert false)
      | _ -> assert false)
    [ x ]

let owens_t_quadrature h a =
  let quad_pts =
    NL.expand_dims
      (T.Concrete
         (Nd.of_floats (Dts.default_float_dtype ()) [| 13 |] c_owens_t_quad_pts))
      (Array.init (ndim a) Fun.id)
  in
  let r = NL.expand_dims (UF.square a) [| ndim a |] *! quad_pts in
  let integrand =
    UF.exp
      (scalar_of h (-0.5)
      *! NL.expand_dims (UF.square h) [| ndim h |]
      *! lit_add 1.0 r)
    /! lit_add 1.0 r
  in
  a
  *! Numpy.Tensor_contractions.matmul integrand
       (T.Concrete
          (Nd.of_floats
             (Dts.default_float_dtype ())
             [| 13 |] c_owens_t_quad_wts))

let owens_t_primal h a =
  let h = UF.abs h in
  let abs_a = UF.abs a in
  let root_2 = lax_const h (sqrt 2.0) in
  let h_normed = h /! root_2 in
  let modified_a = where_ (le_ abs_a 1.0) abs_a (UF.reciprocal abs_a) in
  let modified_h = where_ (le_ abs_a 1.0) h (abs_a *! h) in
  let result = owens_t_quadrature modified_h modified_a in
  let result =
    where_ (eq_ modified_h 0.0)
      (UF.arctan modified_a /! scalar_of h (2.0 *. Float.pi))
      result
  in
  let result =
    where_ (eq_ modified_a 1.0)
      (lit_mul 0.125 (bind1 T.Erfc [ neg_ modified_h /! root_2 ])
      *! bind1 T.Erfc [ modified_h /! root_2 ])
      result
  in
  let normh = bind1 T.Erfc [ h_normed ] in
  let normah = bind1 T.Erfc [ abs_a *! h_normed ] in
  let result =
    where_ (gt_ abs_a 1.0)
      (where_
         (le_ (abs_a *! h) 0.67)
         (lit_sub 0.25
            (lit_mul 0.25 (bind1 T.Erf [ h_normed ])
            *! bind1 T.Erf [ abs_a *! h_normed ])
         -! result)
         (lit_mul 0.25 (normh +! normah -! (normh *! normah)) -! result))
      result
  in
  let result = L.sign a *! result in
  where_
    (bor (UF.isnan a) (UF.isnan h))
    (AC.full_like result Float_format.nan)
    result

let rec owens_t_custom h a =
  List.hd
    (CD.custom_jvp ~name:"_owens_t_impl" ~jvp_name:"_owens_t_jvp"
       (function [ h; a ] -> [ owens_t_primal h a ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ h; a ], [ dh; da ] ->
             let result = owens_t_custom h a in
             let root_2 = lax_const h (sqrt 2.0) in
             let rounded =
               Nd.get_f
                 (Nd.of_floats (dtype h) [||] [| sqrt (2.0 *. Float.pi) |])
                 [||]
             in
             let dout_dh =
               neg_ (L.exp (lit_mul (-0.5) (L.square h)))
               *! bind1 T.Erf [ a *! h /! root_2 ]
               /! lax_const h (2.0 *. rounded)
             in
             let sq_a1 () = add_lit (L.square a) 1.0 in
             let dout_da =
               L.exp (lit_mul (-0.5) (sq_a1 ()) *! L.square h)
               /! (scalar_of h (2.0 *. Float.pi) *! sq_a1 ())
             in
             ( [ result ],
               [ NL.astype ((dout_dh *! dh) +! (dout_da *! da)) (dtype result) ]
             )
         | _ -> assert false)
       [ h; a ])

let owens_t h a =
  let h, a = promote2 h a in
  owens_t_custom h a

let bernoulli n =
  if n < 0 then invalid_arg "n must be a non-negative integer.";
  let dt = Dts.default_float_dtype () in
  let b3 = T.Concrete (Nd.of_floats dt [| 3 |] [| 1.0; -0.5; 1.0 /. 6.0 |]) in
  let module NI = Numpy.Nd_index in
  if n < 3 then NI.take b3 [ NI.Range { start = None; stop = Some (n + 1) } ]
  else begin
    let dyn v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
    let bn =
      NI.set
        (AC.zeros ~dtype:dt [| n + 1 |])
        [ NI.Range { start = None; stop = Some 3 } ]
        b3
    in
    let m = NL.arange ~start:4.0 ~step:2.0 ~dtype:dt (float_of_int (n + 1)) in
    let pi2 = Float.pi ** 2.0 in
    let term = neg_ (sub_lit m 1.0) *! m /! dyn 4.0 /! dyn pi2 in
    let q1 = dyn (1.0 /. pi2) *! RED.cumprod term in
    let k = NL.arange ~start:2.0 ~dtype:dt 50.0 in
    let q2 =
      RED.sum ~axis:[| 0 |]
        (UF.power (NL.expand_dims k [| 1 |]) (neg_ (NL.expand_dims m [| 0 |])))
    in
    NI.set_stepped bn ~start:4 ~step:2 (q1 *! lit_add 1.0 q2)
  end

let while3 cond body init =
  L.while_loop
    (function [ a; b; c ] -> cond a b c | _ -> failwith "while: arity")
    (function [ a; b; c ] -> body a b c | _ -> failwith "while: arity")
    init

let hyp_cond prec k term serie =
  let in_range = UF.less k (py_int 250) in
  let abs_term = L.abs term in
  let ratio = L.div abs_term (L.abs serie) in
  band in_range (UF.greater ratio prec)

let hyp1f1_serie a b x =
  let prec = scalar_of x (eps_of x) in
  let init = [ scalar_of x 1.0; py_int 1; a /! b *! x ] in
  List.hd
    (while3
       (fun serie k term -> hyp_cond prec k term serie)
       (fun serie k term ->
         let serie = serie +! term in
         let term = term *! ((a +! k) /! (b +! k) *! x /! (k +! py_int 1)) in
         [ serie; k +! py_int 1; term ])
       init)

let hyp1f1_asymptotic a b x =
  let prec = scalar_of x (eps_of x) in
  let init = [ scalar_of x 1.0; py_int 1; (b -! a) *! lit_sub 1.0 a /! x ] in
  let serie =
    List.hd
      (while3
         (fun serie k term -> hyp_cond prec k term serie)
         (fun serie k term ->
           let serie = serie +! term in
           let term =
             term
             *! ((b -! a +! k) *! (lit_sub 1.0 a +! k) /! (k +! py_int 1) /! x)
           in
           [ serie; k +! py_int 1; term ])
         init)
  in
  gamma b /! gamma a *! L.exp x *! UF.power x (a -! b) *! serie

let hyp1f1_value a b x =
  let a, b, x = real_only3 "hyp1f1" a b x in
  let result =
    List.hd
      (L.cond
         (UF.less (L.abs x) (scalar_of x 100.0))
         (function [ a; b; x ] -> [ hyp1f1_serie a b x ] | _ -> assert false)
         (function
           | [ a; b; x ] -> [ hyp1f1_asymptotic a b x ] | _ -> assert false)
         [ a; b; x ])
  in
  let to_i v = v *! py_int 1 in
  let index =
    to_i (eq_ a 0.0)
    +! (band (UF.equal a b) (ne_ a 0.0) *! py_int 2)
    +! (band (eq_ b 0.0) (ne_ a 0.0) *! py_int 3)
  in
  L.select_n index
    [ result; scalar_of x 1.0; UF.exp x; scalar_of x Float.infinity ]

let hyp1f1_deriv shifted a b x =
  let prec = scalar_of x (eps_of x) in
  let init = [ scalar_of x 0.0; py_int 1; a /! b *! x ] in
  List.hd
    (while3
       (fun serie k term -> hyp_cond prec k term serie)
       (fun serie k term ->
         let serie = serie +! (term *! shifted k) in
         let term = term *! ((a +! k) /! (b +! k) *! x /! (k +! py_int 1)) in
         [ serie; k +! py_int 1; term ])
       init)

let vectorize3 f =
  vectorize_scalar (function
    | [ a; b; x ] -> [ f a b x ]
    | _ -> failwith "vectorize: arity")

let hyp1f1_jit args =
  List.hd
    (Eager_jit.many ~jit:"hyp1f1" "scipy.special.hyp1f1" () [] args
       (fun _ vals -> vectorize3 hyp1f1_value vals))

let rec hyp1f1_apply a b x =
  List.hd
    (CD.custom_jvp ~name:"hyp1f1"
       (fun args -> [ hyp1f1_jit args ])
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ a; b; x ], [ da; db; dx ] ->
             let out = [ hyp1f1_apply a b x ] in
             let wrt_a =
               List.hd
                 (Eager_jit.many ~jit:"_hyp1f1_a_derivative"
                    "scipy.special._hyp1f1_a_derivative" () [] primals
                    (fun _ vals ->
                      vectorize3
                        (fun a b x ->
                          hyp1f1_deriv
                            (fun k -> digamma (a +! k) -! digamma a)
                            a b x)
                        vals))
               *! da
             in
             let wrt_b =
               List.hd
                 (Eager_jit.many ~jit:"_hyp1f1_b_derivative"
                    "scipy.special._hyp1f1_b_derivative" () [] primals
                    (fun _ vals ->
                      vectorize3
                        (fun a b x ->
                          hyp1f1_deriv
                            (fun k -> digamma b -! digamma (b +! k))
                            a b x)
                        vals))
               *! db
             in
             let wrt_x =
               JU.jit1 "_hyp1f1_x_derivative"
                 (function
                   | [ a; b; x ] ->
                       a /! b *! hyp1f1_apply (add_lit a 1.0) (add_lit b 1.0) x
                   | _ -> assert false)
                 primals
               *! dx
             in
             (out, [ Ad_util.add_any (Ad_util.add_any wrt_a wrt_b) wrt_x ])
         | _ -> failwith "hyp1f1: arity")
       [ a; b; x ])

let hyp1f1 a b x = hyp1f1_apply a b x
let fori_float lower upper body init = L.fori_loop lower upper body init

let expn1 x n =
  let c v = scalar_of x v in
  let cn v = scalar_of n v in
  let machep = eps_of x in
  let zero = c 0.0 and one = c 1.0 in
  let psi0 = c (-.euler_gamma) -! UF.log x in
  let psi =
    List.hd
      (fori_float (cn 1.0) n
         (fun i state ->
           match state with
           | [ psi ] -> [ psi +! (one /! i) ]
           | _ -> assert false)
         [ psi0 ])
  in
  let n1 = where_ (UF.equal n (cn 1.0)) (c 2.0) n in
  let z = neg_ x in
  let pk0 = one -! n in
  let ans0 = where_ (UF.equal n (cn 1.0)) zero (one /! (one -! n1)) in
  let init = [ ans0; pk0; py_float Float.infinity; x; zero; one; z ] in
  let d =
    L.while_loop
      (function
        | [ _; _; t; x; _; _; _ ] ->
            let positive = UF.greater x (c 0.0) in
            band positive (UF.greater t (c machep))
        | _ -> assert false)
      (function
        | [ ans; pk; _; x; xk; yk; z ] ->
            let xk = xk +! one in
            let yk = yk *! (z /! xk) in
            let pk = pk +! one in
            let ans = ans +! where_ (UF.not_equal pk zero) (yk /! pk) zero in
            let t = where_ (UF.not_equal ans zero) (UF.abs (yk /! ans)) one in
            [ ans; pk; t; x; xk; yk; z ]
        | _ -> assert false)
      init
  in
  let ans = List.hd d in
  let z = List.nth d 6 in
  let t = n in
  let r = n -! cn 1.0 in
  (UF.power z r *! psi /! UF.exp (gammaln t)) -! ans

let expn2 x n =
  let c v = scalar_of x v in
  let cn v = scalar_of n v in
  let big = c 1.4411518807585587e+17 in
  let machep = eps_of x in
  let zero = c 0.0 and one = c 1.0 in
  let qkm1 = x +! n in
  let ans0 = one /! (x +! n) in
  let init = [ ans0; cn 1.0; one; one; qkm1; x; zero; c Float.infinity; x ] in
  let d =
    L.while_loop
      (function
        | [ _; k; _; _; _; _; _; t; x ] ->
            let positive = UF.greater x (scalar_of k 0.0) in
            band positive (UF.greater t (c machep))
        | _ -> assert false)
      (function
        | [ ans; k; pkm1; pkm2; qkm1; qkm2; r; _; x ] ->
            let ck v = scalar_of k v in
            let k = k +! ck 1.0 in
            let odd = UF.equal (UF.remainder k (ck 2.0)) (ck 1.0) in
            let yk = where_ odd one x in
            let xk =
              where_ odd (n +! ((k -! ck 1.0) /! ck 2.0)) (k /! ck 2.0)
            in
            let pk = (pkm1 *! yk) +! (pkm2 *! xk) in
            let qk = (qkm1 *! yk) +! (qkm2 *! xk) in
            let nz = UF.not_equal qk zero in
            let r = where_ nz (pk /! qk) r in
            let t = where_ nz (UF.abs ((ans -! r) /! r)) one in
            let ans = where_ nz r ans in
            let is_big = UF.greater (UF.abs pk) big in
            let scale v = where_ is_big (v /! big) v in
            let pkm1' = scale pk in
            let pkm2' = scale pkm1 in
            let qkm1' = scale qk in
            let qkm2' = scale qkm1 in
            [ ans; k; pkm1'; pkm2'; qkm1'; qkm2'; r; t; x ]
        | _ -> assert false)
      init
  in
  List.hd d *! UF.exp (neg_ x)

let expn3 x n =
  let c v = scalar_of x v in
  let one = c 1.0 in
  let xk = x +! n in
  let yk = one /! (xk *! xk) in
  let t = n in
  let ans = yk *! t *! ((c 6.0 *! x *! x) -! (c 8.0 *! t *! x) +! (t *! t)) in
  let ans = yk *! (ans +! (t *! (t -! (c 2.0 *! x)))) in
  let ans = yk *! (ans +! t) in
  (ans +! one) *! UF.exp (neg_ x) /! xk

let expn_core n x =
  let n, x = promote2 n x in
  reject_complex "expn" x;
  let c v = scalar_of x v in
  let cn v = scalar_of n v in
  let zero = c 0.0 and one = c 1.0 in
  let conds =
    [
      bor (UF.less n (cn 0.0)) (UF.less x zero);
      band (UF.equal x zero) (UF.less n (cn 2.0));
      band (UF.equal x zero) (UF.greater_equal n (cn 2.0));
      band (UF.equal n (cn 0.0)) (UF.greater_equal x zero);
      UF.greater_equal n (cn 5000.0);
      UF.greater x one;
    ]
  in
  let n1 = where_ (UF.equal n (cn 1.0)) (n +! n) n in
  let v3 = one /! n1 and v4 = UF.exp (neg_ x) /! x in
  let with_n f = Piece_fn (function [ x; n ] -> f x n | _ -> assert false) in
  piecewise ~key:"expn" ~extra:[ n ] x conds
    [
      constf Float_format.nan;
      constf Float.infinity;
      const_val v3;
      const_val v4;
      with_n expn3;
      with_n expn2;
      with_n expn1;
    ]

let expn_vect args =
  vectorize_scalar
    (fun vals ->
      Eager_jit.many ~jit:"expn" "scipy.special.expn" () [] vals (fun _ vals ->
          match vals with [ n; x ] -> [ expn_core n x ] | _ -> assert false))
    args

let rec expn_apply n x =
  let jvp_body n x x_dot =
    match
      JU.jit "expn_jvp"
        (function
          | [ n; x; x_dot ] ->
              let primal = expn_apply n x in
              let rest = expn_apply (L.sub n (scalar_of n 1.0)) x in
              [ primal; L.mul (L.neg x_dot) rest ]
          | _ -> failwith "expn: arity")
        [ n; x; x_dot ]
    with
    | [ p; t ] -> ([ p ], [ t ])
    | _ -> failwith "expn: arity"
  in
  match n with
  | T.Concrete _ ->
      List.hd
        (CD.custom_jvp ~name:"expn" ~jvp_name:"expn_jvp"
           (function
             | [ x ] -> expn_vect [ n; x ] | _ -> failwith "expn: arity")
           ~jvp:(fun primals tangents ->
             match (primals, tangents) with
             | [ x ], [ x_dot ] -> jvp_body n x x_dot
             | _ -> failwith "expn: arity")
           [ x ])
  | _ ->
      let n = CD.stop_gradient n in
      List.hd
        (CD.custom_jvp ~name:"expn" ~jvp_name:"expn_jvp" expn_vect
           ~jvp:(fun primals tangents ->
             match (primals, tangents) with
             | [ n; x ], [ _; x_dot ] -> jvp_body n x x_dot
             | _ -> failwith "expn: arity")
           [ n; x ])

let expn n x = expn_apply n x

let exp1 x =
  let x = real_only "exp1" x in
  expn_apply (py_int 1) x

let expint_tail a b x =
  let one = scalar_of x 1.0 in
  let w = one /! x in
  let f = polyval (coeff_of x a) w /! polyval (coeff_of x b) w in
  let f = (w *! f) +! one in
  UF.exp x *! w *! f

let expint1 x =
  let f =
    polyval (coeff_of x c_expint1_a) x /! polyval (coeff_of x c_expint1_b) x
  in
  add_lit (x *! f) euler_gamma +! UF.log x

let expi_pos x =
  let range lo hi =
    band (UF.greater x (scalar_of x lo)) (UF.less_equal x (scalar_of x hi))
  in
  let conds =
    range 0.0 2.0
    :: List.map
         (fun i ->
           range (Float.of_int (1 lsl i)) (Float.of_int (1 lsl (i + 1))))
         [ 1; 2; 3; 4; 5 ]
  in
  piecewise ~key:"expi_pos" x conds
    [
      piece_fn expint1;
      piece_fn (expint_tail c_expint2_a c_expint2_b);
      piece_fn (expint_tail c_expint3_a c_expint3_b);
      piece_fn (expint_tail c_expint4_a c_expint4_b);
      piece_fn (expint_tail c_expint5_a c_expint5_b);
      piece_fn (expint_tail c_expint6_a c_expint6_b);
      piece_fn (expint_tail c_expint7_a c_expint7_b);
    ]

let expi_neg x = neg_ (exp1 (neg_ x))

let expi_primal x =
  let x = real_only "expi" x in
  piecewise ~key:"expi" x [ lt_ x 0.0 ] [ piece_fn expi_neg; piece_fn expi_pos ]

let expi_jit a0 =
  Eager_jit.one ~jit:"expi" "scipy.special.expi" () [] [ a0 ] (fun _ vals ->
      match vals with [ a0 ] -> expi_primal a0 | _ -> assert false)

let rec expi x =
  List.hd
    (CD.custom_jvp ~name:"expi" ~jvp_name:"expi_jvp"
       (function [ x ] -> [ expi_jit x ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ x ], [ x_dot ] -> (
             match
               JU.jit "expi_jvp"
                 (function
                   | [ x; x_dot ] -> [ expi x; UF.exp x /! x *! x_dot ]
                   | _ -> assert false)
                 [ x; x_dot ]
             with
             | [ p; t ] -> ([ p ], [ t ])
             | _ -> assert false)
         | _ -> assert false)
       [ x ])

let sp_is_integral c = eq_ (UF.remainder c (py_int 1)) 0.0
let sp_and = UF.logical_and
let sp_or = UF.logical_or
let sp_not = UF.logical_not

let hyp2f1_terminal a b c x =
  let eps = eps50 x in
  let ib = NL.round b in
  let mask =
    sp_and (UF.less b a)
      (sp_and
         (UF.less (UF.abs (b -! ib)) eps)
         (sp_not
            (sp_and (sp_is_integral c) (sp_and (le_ c 0.0) (UF.greater c b)))))
  in
  let orig_a = a in
  let a = where_ mask b a in
  let b = where_ mask orig_a b in
  let a = UF.abs a in
  let one_x = scalar_of x 1.0 in
  let looped =
    L.fori_loop (scalar_of a 1.0) (add_lit a 1.0)
      (fun i state ->
        match state with
        | [ serie; term ] ->
            let one = scalar_of i 1.0 in
            let term =
              term
              *! (neg_ (a -! i +! one)
                 /! (c +! i -! one)
                 *! (b +! i -! one)
                 /! i *! x)
            in
            [ serie +! term; term ]
        | _ -> assert false)
      [ one_x; one_x ]
  in
  List.hd looped

let hyp2f1_serie a b c x =
  let rtol = scalar_of x (eps_of x) in
  List.hd
    (while3
       (fun serie k term ->
         band (lt_ k 250.0) (UF.greater (L.abs term) (rtol *! L.abs serie)))
       (fun serie k term ->
         let serie = serie +! term in
         let one = scalar_of k 1.0 in
         let term =
           term
           *! ((a +! k -! one) *! (b +! k -! one) /! (c +! k -! one) /! k *! x)
         in
         [ serie; k +! one; term ])
       [ scalar_of x 0.0; scalar_of x 1.0; scalar_of x 1.0 ])

let hyp2f1_digamma_transform a b c x =
  let rtol = scalar_of x (eps_of x) in
  let d = c -! a -! b in
  let s = lit_sub 1.0 x in
  let rd = NL.round d in
  let e = where_ (ge_ rd 0.0) d (neg_ d) in
  let d1 = where_ (ge_ rd 0.0) d (scalar_of d 0.0) in
  let d2 = where_ (ge_ rd 0.0) (scalar_of d 0.0) d in
  let ard = NL.astype (where_ (ge_ rd 0.0) rd (neg_ rd)) D.I32 in
  let ax = UF.log s in
  let dg v = digamma v in
  let y0 = dg (JU.py_float 1.0) in
  let y1 = dg (lit_add 1.0 e) in
  let y = y0 +! y1 in
  let y2 = dg (a +! d1) in
  let y = y -! y2 in
  let y3 = dg (b +! d1) in
  let y = y -! y3 in
  let y = y -! ax in
  let y = y /! gamma (add_lit e 1.0) in
  let p0 = (a +! d1) *! (b +! d1) in
  let p0 = p0 *! s in
  let p = p0 /! gamma (add_lit e 2.0) in
  let looped =
    L.while_loop
      (function
        | [ _; _; _; _; _; _; q; _; _; t; y ] ->
            let keep = lt_ t 250.0 in
            let aq = UF.abs q in
            sp_and keep (UF.greater_equal aq (rtol *! UF.abs y))
        | _ -> assert false)
      (function
        | [ a; ax; b; d1; e; p; _; _; s; t; y ] ->
            let r0 = dg (lit_add 1.0 t) in
            let r1 = dg (lit_add 1.0 t +! e) in
            let r = r0 +! r1 in
            let r2 = dg (a +! t +! d1) in
            let r = r -! r2 in
            let r3 = dg (b +! t +! d1) in
            let r = r -! r3 in
            let r = r -! ax in
            let q = p *! r in
            let y = y +! q in
            let f0 = s *! (a +! t +! d1) in
            let f0 = f0 /! add_lit t 1.0 in
            let p = p *! f0 in
            let f1 = b +! t +! d1 in
            let f1 = f1 /! (add_lit t 1.0 +! e) in
            let p = p *! f1 in
            let t = add_lit t 1.0 in
            [ a; ax; b; d1; e; p; q; r; s; t; y ]
        | _ -> assert false)
      [ a; ax; b; d1; e; p; y; scalar_of x 0.0; s; scalar_of x 1.0; y ]
  in
  let y = List.nth looped 10 in
  let compute_sum y =
    let y1_init = scalar_of x 1.0 in
    let t0 = scalar_of x 0.0 in
    let p0 = scalar_of x 1.0 in
    let looped =
      L.fori_loop (int_full D.I32 1) ard
        (fun _ state ->
          match state with
          | [ a; b; d2; e; p; s; t; y1 ] ->
              let r = lit_sub 1.0 e +! t in
              let f = s *! (a +! t +! d2) in
              let f = f *! (b +! t +! d2) in
              let f = f /! r in
              let p = p *! f in
              let t = add_lit t 1.0 in
              let p = p /! t in
              let y1 = y1 +! p in
              [ a; b; d2; e; p; s; t; y1 ]
          | _ -> assert false)
        [ a; b; d2; e; p0; s; t0; y1_init ]
    in
    let y1 = List.nth looped 7 in
    let p = gamma c in
    let g_e = gamma e in
    let top = g_e *! p in
    let g_a = gamma (a +! d1) in
    let g_b = gamma (b +! d1) in
    let bottom = g_a *! g_b in
    let y1 = y1 *! (top /! bottom) in
    let g_a2 = gamma (a +! d2) in
    let g_b2 = gamma (b +! d2) in
    let bottom2 = g_a2 *! g_b2 in
    let y = y *! (p /! bottom2) in
    let y = where_ (ne_ (band ard (int_full D.I32 1)) 0.0) (neg_ y) y in
    let q = UF.power s rd in
    let positive = gt_ rd 0.0 in
    let when_pos = (y *! q) +! y1 in
    let when_neg = y +! (y1 *! q) in
    where_ positive when_pos when_neg
  in
  let is_zero = eq_ rd 0.0 in
  let g_c = gamma c in
  let top = y *! g_c in
  let g_a = gamma a in
  let g_b = gamma b in
  let bottom = g_a *! g_b in
  let at_zero = top /! bottom in
  let summed = compute_sum y in
  where_ is_zero at_zero summed

let hyp2f1_terminal_or_serie a b c x =
  let eps = eps50 x in
  let d = c -! a -! b in
  let ia = NL.round a and ib = NL.round b and id = NL.round d in
  let neg_int_a = sp_and (le_ a 0.0) (UF.less (UF.abs (a -! ia)) eps) in
  let neg_int_b = sp_and (le_ b 0.0) (UF.less (UF.abs (b -! ib)) eps) in
  let either = sp_or neg_int_a neg_int_b in
  let not_either = sp_not either in
  let index =
    where_
      (sp_and (gt_ x 0.9) not_either)
      (where_ (UF.greater_equal (UF.abs (d -! id)) eps) (py_int 0) (py_int 1))
      (where_ either (py_int 2) (py_int 0))
  in
  let serie = hyp2f1_serie a b c x in
  let digamma_branch = hyp2f1_digamma_transform a b c x in
  let terminal = hyp2f1_terminal a b c x in
  L.select_n index [ serie; digamma_branch; terminal ]

let hyp2f1_value a b c x =
  let a, b, c, x =
    match promote_inexact [ a; b; c; x ] with
    | [ a; b; c; x ] -> (a, b, c, x)
    | _ -> assert false
  in
  let eps = eps50 x in
  let d = c -! a -! b in
  let s = lit_sub 1.0 x in
  let ca = c -! a and cb = c -! b in
  let id = NL.round d in
  let index =
    where_
      (sp_or (eq_ x 0.0) (sp_and (sp_or (eq_ a 0.0) (eq_ b 0.0)) (ne_ c 0.0)))
      (py_int 0)
      (where_
         (sp_or (eq_ c 0.0) (sp_and (lt_ c 0.0) (sp_is_integral c)))
         (py_int 1)
         (where_
            (sp_and (le_ d (-1.0))
               (sp_not
                  (sp_and (UF.greater_equal (UF.abs (d -! id)) eps) (lt_ s 0.0))))
            (py_int 2)
            (where_
               (sp_and (le_ d 0.0) (eq_ x 1.0))
               (py_int 1)
               (where_
                  (sp_and (lt_ x 1.0) (UF.equal b c))
                  (py_int 3)
                  (where_
                     (sp_and (lt_ x 1.0) (UF.equal a c))
                     (py_int 4)
                     (where_ (gt_ x 1.0) (py_int 1)
                        (where_ (eq_ x 1.0) (py_int 5) (py_int 6))))))))
  in
  let one = scalar_of x 1.0 in
  let inf = scalar_of x Float.infinity in
  let pow_d = UF.power s d in
  let first = hyp2f1_terminal_or_serie ca cb c x in
  let branch2 = pow_d *! first in
  let neg_a = neg_ a in
  let branch3 = UF.power s neg_a in
  let neg_b = neg_ b in
  let branch4 = UF.power s neg_b in
  let gc = gamma c in
  let gd = gamma d in
  let top = gc *! gd in
  let gca = gamma ca in
  let gcb = gamma cb in
  let bottom = gca *! gcb in
  let branch5 = top /! bottom in
  let branch6 = hyp2f1_terminal_or_serie a b c x in
  L.select_n index [ one; inf; branch2; branch3; branch4; branch5; branch6 ]

let vectorize4 f =
  vectorize_scalar (function
    | [ a; b; c; x ] -> [ f a b c x ]
    | _ -> failwith "vectorize: arity")

let hyp2f1_deriv shifted a b c x =
  let prec = scalar_of x (eps_of x) in
  let weak_int v = C.py_scalar (default_int ()) v in
  let ab = a *! b in
  let abc = ab /! c in
  let init = [ weak_int 0.0; weak_int 1.0; abc *! x ] in
  List.hd
    (while3
       (fun serie k term -> hyp_cond prec k term serie)
       (fun serie k term ->
         let d = shifted k in
         let added = term *! d in
         let serie = serie +! added in
         let ak = a +! k in
         let bk = b +! k in
         let f = ak *! bk in
         let ck = c +! k in
         let f = f /! ck in
         let k1 = k +! weak_int 1.0 in
         let f = f /! k1 in
         let f = f *! x in
         let term = term *! f in
         [ serie; k +! weak_int 1.0; term ])
       init)

let rec hyp2f1_apply a b c x =
  let primal args =
    Eager_jit.many ~jit:"hyp2f1" "scipy.special.hyp2f1" () [] args
      (fun _ vals -> vectorize4 hyp2f1_value vals)
  in
  let deriv name shifted args =
    List.hd
      (Eager_jit.many ~jit:name ("scipy.special." ^ name) () [] args
         (fun _ vals ->
           vectorize4 (fun a b c x -> hyp2f1_deriv (shifted a b c) a b c x) vals))
  in
  let rule name shifted =
    Some (fun t _ primals -> [ deriv name shifted primals *! t ])
  in
  let wrt_x t _ primals =
    let d =
      JU.jit1 "_hyp2f1_x_derivative"
        (function
          | [ a; b; c; x ] ->
              let ratio = a *! b /! c in
              let a1 = a +! py_float 1.0 in
              let b1 = b +! py_float 1.0 in
              let c1 = c +! py_float 1.0 in
              ratio *! hyp2f1_apply a1 b1 c1 x
          | _ -> assert false)
        primals
    in
    [ d *! t ]
  in
  List.hd
    ((CD.defjvps ~name:"hyp2f1" primal
        [
          rule "_hyp2f1_a_derivative" (fun a _ _ k ->
              let top = digamma (a +! k) in
              top -! digamma a);
          rule "_hyp2f1_b_derivative" (fun _ b _ k ->
              let top = digamma (b +! k) in
              top -! digamma b);
          rule "_hyp2f1_c_derivative" (fun _ _ c k ->
              let top = digamma c in
              top -! digamma (c +! k));
          Some wrt_x;
        ])
       [ a; b; c; x ])

let hyp2f1 a b c x = hyp2f1_apply a b c x

let sincospisquaredhalf x =
  let x = UF.abs x in
  let s = UF.fmod x (py_float 2.0) in
  let r = UF.fmod (s *! (x -! div_lit s 2.0)) (py_float 2.0) in
  let pi = scalar_of x Float.pi in
  let sinpi =
    where_ (lt_ r 0.5)
      (UF.sin (pi *! r))
      (where_ (gt_ r 1.5)
         (UF.sin (pi *! sub_lit r 2.0))
         (neg_ (UF.sin (pi *! sub_lit r 1.0))))
  in
  let cospi =
    where_px (eq_ r 0.5) 0.0
      (where_ (lt_ r 1.0)
         (neg_ (UF.sin (pi *! sub_lit r 0.5)))
         (UF.sin (pi *! sub_lit r 1.5)))
  in
  (sinpi, cospi)

let sincospisquaredhalf_jit a0 =
  Eager_jit.pair ~jit:"sincospisquaredhalf" "scipy.special.sincospisquaredhalf"
    () [] [ a0 ] (fun _ vals ->
      match vals with [ a0 ] -> sincospisquaredhalf a0 | _ -> assert false)

let fresnel_primal xxa0 =
  let xxa = promote1 xxa0 in
  let original_dtype = dtype xxa in
  if is_complex_dtype original_dtype then
    failwith "Support for complex-valued inputs is not implemented yet.";
  let xxa =
    match original_dtype with
    | D.F32 | D.F16 | D.Bf16 -> NL.astype xxa D.F32
    | D.F64 -> xxa
    | _ -> failwith "Support for this dtype is not implemented yet."
  in
  let single = dtype xxa = D.F32 in
  if single then
    for _ = 1 to 4 do
      ignore (AC.empty [| 0 |])
    done;
  let x = UF.abs xxa in
  let x2 = x *! x in
  let t = x2 *! x2 in
  let cv = coeff_of x in
  let s_small, c_small =
    if single then
      ( x *! x2 *! polyval (cv c_fresnl_sn_f32) t,
        x *! polyval (cv c_fresnl_cn_f32) t )
    else
      ( x *! x2
        *! polyval (cv c_fresnl_sn_f64) t
        /! polyval (cv c_fresnl_sd_f64) t,
        x *! polyval (cv c_fresnl_cn_f64) t /! polyval (cv c_fresnl_cd_f64) t )
  in
  let sinpi, cospi = sincospisquaredhalf_jit x in
  let pi = scalar_of x Float.pi in
  let large_parts =
    if single then None
    else
      let c_large = lit_add 0.5 (lit_div 1.0 (pi *! x) *! sinpi) in
      let s_large = lit_sub 0.5 (lit_div 1.0 (pi *! x) *! cospi) in
      Some (c_large, s_large)
  in
  let tt = pi *! x2 in
  let u = lit_div 1.0 (tt *! tt) in
  let tt = lit_div 1.0 tt in
  let f, g =
    if single then
      ( lit_sub 1.0 (u *! polyval (cv c_fresnl_fn_f32) u),
        tt *! polyval (cv c_fresnl_gn_f32) u )
    else
      ( lit_sub 1.0
          (u *! polyval (cv c_fresnl_fn_f64) u /! polyval (cv c_fresnl_fd_f64) u),
        tt *! polyval (cv c_fresnl_gn_f64) u /! polyval (cv c_fresnl_gd_f64) u
      )
  in
  let t3 = pi *! x in
  let c_other = lit_add 0.5 (((f *! sinpi) -! (g *! cospi)) /! t3) in
  let s_other = lit_sub 0.5 (((f *! cospi) +! (g *! sinpi)) /! t3) in
  let isinf = UF.isinf xxa in
  let small = lt_ x2 2.5625 in
  let large = gt_ x 36974.0 in
  let choose small_part large_part other =
    let far =
      match large_part with
      | None -> where_px large 0.5 other
      | Some v -> where_ large v other
    in
    where_px isinf 0.5 (where_ small small_part far)
  in
  let s = choose s_small (Option.map snd large_parts) s_other in
  let c = choose c_small (Option.map fst large_parts) c_other in
  let neg = lt_ xxa 0.0 in
  let s = where_ neg (neg_ s) s in
  let c = where_ neg (neg_ c) c in
  if original_dtype <> dtype xxa then
    [ NL.astype s original_dtype; NL.astype c original_dtype ]
  else [ s; c ]

let rec fresnel x =
  match
    CD.custom_jvp ~name:"fresnel" ~jvp_name:"_fresnel_jvp"
      (function [ x ] -> fresnel_primal x | _ -> assert false)
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ x ], [ x_dot ] ->
            let s, c = fresnel x in
            let result = [ s; c ] in
            let sinpi, cospi = sincospisquaredhalf_jit x in
            (result, [ sinpi *! x_dot; cospi *! x_dot ])
        | _ -> assert false)
      [ x ]
  with
  | [ s; c ] -> (s, c)
  | _ -> invalid_arg "fresnel: arity"

let full_like v x = const_full (dtype v) (shape v) x

let bessel_jn_one ~v ~n_iter z =
  let dt = dtype z in
  let f0 = scalar_of z 0.0 in
  let f1 = scalar_of z 1e-16 in
  let bs = scalar_of z 0.0 in
  let body args =
    match args with
    | [ f0; f1; bs; z; k ] ->
        let num = lit_mul 2.0 (add_lit k 1.0) *! f1 in
        let f = (num /! z) -! f0 in
        let cond_pred = eq_ (UF.remainder k (scalar_of k 2.0)) 0.0 in
        let bs =
          List.hd
            (L.cond cond_pred
               (function
                 | [ bs; f ] -> [ bs +! lit_mul 2.0 f ] | _ -> assert false)
               (function [ bs; _ ] -> [ bs ] | _ -> assert false)
               [ bs; f ])
        in
        [ f1; f; bs; z; f ]
    | _ -> assert false
  in
  let out =
    L.scan ~reverse:true body [ f0; f1; bs; z ] [ L.iota dt (n_iter + 1) ]
  in
  match out with
  | [ _; _; bs; _; j_vals ] ->
      let first = L.index_in_dim ~keepdims:false j_vals 0 in
      let picked = L.slice_in_dim j_vals 0 (v + 1) in
      picked /! (bs -! first)
  | _ -> assert false

let bessel_jn ?(n_iter = 50) ~v z =
  let z = promote1 z in
  reject_complex "bessel_jn" z;
  let rank = ndim z in
  let rec wrap d f =
    if d = 0 then f
    else wrap (d - 1) (fun xs -> Interpreters.Batching.vmap f [ Some 0 ] xs)
  in
  let out =
    List.hd
      (wrap rank
         (function
           | [ z ] -> [ bessel_jn_one ~v ~n_iter z ] | _ -> assert false)
         [ z ])
  in
  NL.moveaxis [| -1 |] [| 0 |] out

let einsum =
  Numpy.Einsum.einsum ?optimize:None ?precision:None
    ?preferred_element_type:None

let arange_dt dt lo hi =
  NL.arange ~start:(float_of_int lo) ~dtype:dt (float_of_int hi)

let meshgrid_ij dt n m =
  let first = arange_dt dt 0 n in
  let second = arange_dt dt 0 m in
  match NL.meshgrid ~indexing:"ij" [ first; second ] with
  | [ a; b ] -> (a, b)
  | _ -> assert false

module Ix = Numpy.Nd_index

let triu_size n m k =
  let count = ref 0 in
  for i = 0 to n - 1 do
    for j = 0 to m - 1 do
      if j - i >= k then incr count
    done
  done;
  !count

let triu_index_arrays ?m n k =
  let m = Option.value m ~default:n in
  let ones = AC.ones [| n; m |] in
  let upper = NL.triu ~k ones in
  match Numpy.Extras.nonzero ~size:(triu_size n m k) upper with
  | [ i; j ] -> [ Ix.Arr i; Ix.Arr j ]
  | _ -> assert false

let py_weak_int v = C.py_scalar (default_int ()) (float_of_int v)

let gen_recurrence_mask l_max is_normalized dt =
  let n = l_max + 1 in
  let m_mat, l_mat = meshgrid_ij dt n n in
  let lit v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
  let d0, d1 =
    if is_normalized then begin
      let c0 = l_mat *! l_mat in
      let c1 = m_mat *! m_mat in
      let c2 = lit 2.0 *! l_mat in
      let lm1 = l_mat -! lit 1.0 in
      let c3 = lm1 *! (l_mat -! lit 1.0) in
      let n0 = (lit 4.0 *! c0) -! lit 1.0 in
      let d0 = UF.sqrt (n0 /! (c0 -! c1)) in
      let n1 = (c2 +! lit 1.0) *! (c3 -! c1) in
      let den = (c2 -! lit 3.0) *! (c0 -! c1) in
      let d1 = UF.sqrt (n1 /! den) in
      (d0, d1)
    end
    else begin
      let n0 = (lit 2.0 *! l_mat) -! lit 1.0 in
      let d0 = n0 /! (l_mat -! m_mat) in
      let n1 = l_mat +! m_mat -! lit 1.0 in
      let d1 = n1 /! (l_mat -! m_mat) in
      (d0, d1)
    end
  in
  let d0_idx = triu_index_arrays n 1 in
  let d1_idx = triu_index_arrays n 2 in
  let d_zeros = AC.zeros ~dtype:dt [| n; n |] in
  let d0_mask = Ix.set d_zeros d0_idx (Ix.take d0 d0_idx) in
  let d1_mask = Ix.set d_zeros d1_idx (Ix.take d1 d1_idx) in
  let range = Numpy.Grids.Step (0.0, float_of_int n, 1.0) in
  let mask =
    match Numpy.Grids.ogrid [ range; range; range ] with
    | [ i; j; k ] -> NL.astype (UF.equal (i +! j -! k) (py_weak_int 0)) dt
    | _ -> assert false
  in
  let d0_3d = einsum "jk,ijk->ijk" [ d0_mask; mask ] in
  let d1_3d = einsum "jk,ijk->ijk" [ d1_mask; mask ] in
  (d0_3d, d1_3d)

let gen_associated_legendre_body l_max x is_normalized =
  let dt = dtype x in
  let lit v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
  let n = (shape x).(0) in
  let p = AC.zeros ~dtype:dt [| l_max + 1; l_max + 1; n |] in
  let a_idx = arange_dt dt 1 (l_max + 1) in
  let b_idx = arange_dt dt 0 l_max in
  let initial_value, f_a, f_b =
    if is_normalized then begin
      let root = UF.sqrt (py_float Float.pi) in
      let initial = py_float 0.5 /! root in
      let inner = UF.sqrt (lit 1.0 +! (lit 0.5 /! a_idx)) in
      let f_a = RED.cumprod ~axis:0 (lit (-1.0) *! inner) in
      let f_b = UF.sqrt ((lit 2.0 *! b_idx) +! lit 3.0) in
      (initial, f_a, f_b)
    end
    else
      let f_a = RED.cumprod ~axis:0 (lit 1.0 -! (lit 2.0 *! a_idx)) in
      let f_b = (lit 2.0 *! b_idx) +! lit 1.0 in
      (py_float 1.0, f_a, f_b)
  in
  let p = Ix.set p [ Ix.Int 0; Ix.Int 0 ] initial_value in
  let root = UF.sqrt (lit 1.0 -! (x *! x)) in
  let y = RED.cumprod ~axis:0 (NL.broadcast_to root [| l_max; n |]) in
  let e1 = einsum "i,ij->ij" [ f_a; y ] in
  let p_diag = initial_value *! e1 in
  let diag = List.hd (NL.diag_indices (l_max + 1)) in
  let tail = Ix.take diag [ Ix.Range { start = Some 1; stop = None } ] in
  let tail2 = Ix.take diag [ Ix.Range { start = Some 1; stop = None } ] in
  let p = Ix.set p [ Ix.Arr tail; Ix.Arr tail2 ] p_diag in
  let fbx = einsum "i,j->ij" [ f_b; x ] in
  let short = List.hd (NL.diag_indices l_max) in
  let on_diag = Ix.take p [ Ix.Arr short; Ix.Arr short ] in
  let p_offdiag = einsum "ij,ij->ij" [ fbx; on_diag ] in
  let head = Ix.take diag [ Ix.Range { start = None; stop = Some l_max } ] in
  let head2 = Ix.take diag [ Ix.Range { start = None; stop = Some l_max } ] in
  let head2 = head2 +! py_weak_int 1 in
  let p = Ix.set p [ Ix.Arr head; Ix.Arr head2 ] p_offdiag in
  let d0_mask_3d, d1_mask_3d = gen_recurrence_mask l_max is_normalized dt in
  if l_max > 1 then
    List.hd
      (L.fori_loop (py_int 2)
         (py_int (l_max + 1))
         (fun i state ->
           match state with
           | [ p_val ] ->
               let coeff_0 = Ix.take d0_mask_3d [ Ix.Arr i ] in
               let coeff_1 = Ix.take d1_mask_3d [ Ix.Arr i ] in
               let r1 = NL.roll ~axis:[| 1 |] p_val [| 1 |] in
               let e1 = einsum "ijk,k->ijk" [ r1; x ] in
               let t1 = einsum "ij,ijk->ijk" [ coeff_0; e1 ] in
               let r2 = NL.roll ~axis:[| 1 |] p_val [| 2 |] in
               let t2 = einsum "ij,ijk->ijk" [ coeff_1; r2 ] in
               let h = t1 -! t2 in
               [ p_val +! h ]
           | _ -> assert false)
         [ p ])
  else p

let gen_associated_legendre l_max x is_normalized =
  Eager_jit.one "scipy.special._gen_associated_legendre" (l_max, is_normalized)
    [] [ x ] (fun _ vals ->
      JU.jit1 "_gen_associated_legendre"
        ~key:(Printf.sprintf "%d_%b" l_max is_normalized)
        (function
          | [ x ] -> gen_associated_legendre_body l_max x is_normalized
          | _ -> assert false)
        vals)

let gen_derivatives_body p x =
  let dt = dtype x in
  let lit v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
  let sh = shape p in
  let num_m = sh.(0) and num_l = sh.(1) in
  let all = Ix.Range { start = None; stop = None } in
  let upto stop = Ix.Range { start = None; stop = Some stop } in
  let between lo hi = Ix.Range { start = Some lo; stop = Some hi } in
  let p_m_lm1 =
    Ix.take (NL.pad p [| (0, 0); (1, 0); (0, 0) |] 0.0) [ all; upto num_l; all ]
  in
  let p_mp2_lm1 =
    Ix.take
      (NL.pad p_m_lm1 [| (0, 2); (0, 0); (0, 0) |] 0.0)
      [ between 2 (num_m + 2); all; all ]
  in
  let p_mm2_lm1 =
    Ix.take
      (NL.pad p_m_lm1 [| (2, 0); (0, 0); (0, 0) |] 0.0)
      [ upto num_m; all; all ]
  in
  let p_mm2_lm1 =
    if num_l > 1 then begin
      let l_vec = arange_dt dt 1 (num_l - 1) in
      let p_p1 = Ix.take p [ Ix.Int 1; between 1 (num_l - 1); all ] in
      let den = (l_vec +! lit 1.0) *! l_vec in
      let coeff = lit (-1.0) /! den in
      let update = einsum "i,ij->ij" [ coeff; p_p1 ] in
      Ix.set p_mm2_lm1 [ Ix.Int 1; between 2 num_l; all ] update
    end
    else p_mm2_lm1
  in
  let p_mm2_lm1 =
    if num_l > 2 then begin
      let l_vec = arange_dt dt 2 (num_l - 1) in
      let p_p2 = Ix.take p [ Ix.Int 2; between 2 (num_l - 1); all ] in
      let d1 = l_vec +! lit 2.0 in
      let d2 = d1 *! (l_vec +! lit 1.0) in
      let d3 = d2 *! l_vec in
      let den = d3 *! (l_vec -! lit 1.0) in
      let coeff = lit 1.0 /! den in
      let update = einsum "i,ij->ij" [ coeff; p_p2 ] in
      Ix.set p_mm2_lm1 [ Ix.Int 0; between 3 num_l; all ] update
    end
    else p_mm2_lm1
  in
  let m_mat, l_mat = meshgrid_ij dt num_m num_l in
  let coeff_zeros = AC.zeros ~dtype:dt [| num_m; num_l |] in
  let upper = triu_index_arrays ~m:num_l num_m 0 in
  let zero_vec = AC.zeros ~dtype:dt [| num_l |] in
  let masked v = Ix.set coeff_zeros upper (Ix.take v upper) in
  let a0 = lit (-0.5) /! (m_mat -! lit 1.0) in
  let a0_masked = masked a0 in
  let a0_masked = Ix.set a0_masked [ Ix.Int 1; all ] zero_vec in
  let b0 = l_mat +! m_mat in
  let c0 = a0 *! (b0 -! lit 2.0) *! (b0 -! lit 1.0) in
  let c0_masked = masked c0 in
  let c0_masked = Ix.set c0_masked [ Ix.Int 1; all ] zero_vec in
  let e1 = einsum "ij,ijk->ijk" [ a0_masked; p_m_lm1 ] in
  let e2 = einsum "ij,ijk->ijk" [ c0_masked; p_mm2_lm1 ] in
  let p_mm1_l = e1 +! e2 in
  let d0 = lit (-0.5) /! (m_mat +! lit 1.0) in
  let d0_masked = masked d0 in
  let e0 = d0 *! b0 *! (b0 +! lit 1.0) in
  let e0_masked = masked e0 in
  let e3 = einsum "ij,ijk->ijk" [ d0_masked; p_mp2_lm1 ] in
  let e4 = einsum "ij,ijk->ijk" [ e0_masked; p_m_lm1 ] in
  let p_mp1_l = e3 +! e4 in
  let lmm = l_mat -! m_mat +! lit 1.0 in
  let f0 = b0 *! lmm /! lit 2.0 in
  let f0_masked = masked f0 in
  let e5 = einsum "ij,ijk->ijk" [ f0_masked; p_mm1_l ] in
  let p_derivative = e5 -! (lit 0.5 *! p_mp1_l) in
  if num_m > 1 then begin
    let l_vec = arange_dt dt 0 num_l in
    let lv = (l_vec +! lit 1.0) *! l_vec in
    let row0 = Ix.take p [ Ix.Int 0; all; all ] in
    let g0 = einsum "i,ij->ij" [ lv; row0 ] in
    let g0 = if num_l > 2 then g0 -! Ix.take p [ Ix.Int 2; all; all ] else g0 in
    let xx = x *! x in
    let inv = lit 0.5 /! UF.sqrt (lit 1.0 -! xx) in
    let m0 = einsum "j,ij->ij" [ inv; g0 ] in
    let pd = Ix.set p_derivative [ Ix.Int 1; all; all ] m0 in
    Ix.set pd [ Ix.Int 1; Ix.Int 0; all ] (py_weak_int 0)
  end
  else p_derivative

let gen_derivatives p x =
  Eager_jit.one "scipy.special._gen_derivatives" () [] [ p; x ] (fun _ vals ->
      JU.jit1 "_gen_derivatives"
        (function [ p; x ] -> gen_derivatives_body p x | _ -> assert false)
        vals)

let check_lpmn m n z =
  (match dtype z with
  | D.F32 | D.F64 -> ()
  | d ->
      invalid_arg
        (Printf.sprintf
           "z.dtype=%s is not supported, see docstring for supported types."
           (D.short_name d)));
  if Array.length (shape z) <> 1 then invalid_arg "z must be a 1D array.";
  if m <> n then invalid_arg "Computations for m!=n are not yet supported."

let lpmn m n z =
  check_lpmn m n z;
  let p_vals = gen_associated_legendre n z false in
  let p_derivatives = gen_derivatives p_vals z in
  [ p_vals; p_derivatives ]

let lpmn_values m n z is_normalized =
  check_lpmn m n z;
  gen_associated_legendre n z is_normalized

let sph_harm_body n_max args =
  match args with
  | [ n; m; theta; phi ] ->
      let cos_colatitude = UF.cos theta in
      let legendre = gen_associated_legendre n_max cos_colatitude true in
      let abs_m = UF.abs m in
      let lanes =
        NL.arange ~dtype:(default_int ()) (float_of_int (shape n).(0))
      in
      let legendre_val =
        Ix.take_clip legendre [ Ix.Arr abs_m; Ix.Arr n; Ix.Arr lanes ]
      in
      let angle = UF.abs m *! phi in
      let cosa = UF.cos angle in
      let sina = UF.sin angle in
      let vandermonde = L.complex cosa sina in
      let re = legendre_val *! UF.real vandermonde in
      let im = legendre_val *! UF.imag vandermonde in
      let harmonics = L.complex re im in
      let negative = UF.less m (py_weak_int 0) in
      let sign = UF.power (py_float (-1.0)) (UF.abs m) in
      let negative_order = sign *! UF.conj harmonics in
      where_ negative negative_order harmonics
  | _ -> assert false

let sph_harm_y ?diff_n ?n_max n m theta phi =
  if diff_n <> None then
    invalid_arg
      "The 'diff_n' argument to jax.scipy.special.sph_harm_y is not supported.";
  let theta =
    if Array.length (shape theta) = 0 then L.broadcast_in_dim theta [| 1 |] [||]
    else theta
  in
  let n_max =
    match n_max with
    | Some v -> v
    | None -> (
        match RED.amax n with
        | T.Concrete nd -> int_of_float (Nd.get_f nd [||])
        | T.Device b -> int_of_float (Nd.get_f (Pjrt.Buffer.to_host b) [||])
        | T.Tracer _ ->
            invalid_arg
              "The `n_max` argument of `jnp.scipy.special.sph_harm` must be \
               statically specified to use `sph_harm` within JAX \
               transformations.")
  in
  Eager_jit.one ~jit:"_sph_harm" "scipy.special._sph_harm" n_max []
    [ n; m; theta; phi ] (fun _ vals -> sph_harm_body n_max vals)

let gammasgn = gammasgn_jit

let bessel_jn ?n_iter ~v a1 =
  Eager_jit.one ~jit:"bessel_jn" "scipy.special.bessel_jn" (n_iter, v) [] [ a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a1 ] -> bessel_jn ?n_iter ~v a1
      | _ -> assert false)

let logsumexp = Ops.Special.logsumexp
let logsumexp_sign = Ops.Special.logsumexp_sign
