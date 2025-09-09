#include <iostream>
#include <math.h>
#include <Eigen/Dense> // https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
#include "matplotlibcpp.h"
#include "SBA_socket.cpp"

#define DT 0.00001

namespace plt = matplotlibcpp;
using Eigen::Matrix2d;
using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::Vector4d;

class Wind_Model {
	private:
		double v_tw;
		double alpha_tw;
	public:
		Wind_Model(double intensity=0, double direction=0){
			v_tw = abs(intensity);
			alpha_tw = direction;
		}
		double get_v_tw(){
			return v_tw;
		}
		double get_alpha_tw(){
            return alpha_tw;
        }
};

class Sailing_Boat_Model {
	private:
		Vector4d eta; // 		x, 		y, 		phi, 	xi
		Vector4d nu; // 		u, 		v, 		p, 		r
		Vector4d eta_dot;
		Vector4d nu_dot;
		// STATE VARIABLE
		double x;
		double y;
		double phi;
		double xi;
		double u;
		double v;
		double p;
		double r;
		double phi_dot;
		double xi_dot;
		// CONSTANTS
		double rho_water = 1025;
		double rho_air = 1.23;
        double a = -19336/*-5.89*/;
        double b = 467534/*8160*/;
        double c = 39393676.2/*12000*/;
        double d = 65656127/*20000*/;
		double m = 25900;
	  	double I_xx = 133690;
	  	double I_zz = 24760;
	  	double I_xz = 2180;
	  	double x_u_dot = -970;
	  	double y_v_dot = -17430;
	  	double k_p_dot = -106500;
	  	double n_r_dot = -101650;
	  	double x_h = 0;
        double y_h = 0;
        double z_h = -1.18;
	  	double A_k = 8.7;
	  	double x_k = 0;
	  	double y_k = 0;
	  	double z_k = -0.58;
	  	double AR_k = 1;
	  	double A_s = 170;
	  	double x_s = 0;
	  	double y_s = 0;
	  	double z_s = -11.58;
	  	double AR_s = 3;
	  	double x_sm = 0.6;
	  	double x_m = 0.3;
	  	double A_r = 1.17;
	  	double x_r = -8.2;
	  	double y_r = 0;
	  	double z_r = -0.78;
	  	double AR_r = 4.94;
	  	double p0_sail_CL = -1.44183;
		double p1_sail_CL = 0.35006;
		double p2_sail_CL = -0.0410826;
		double p3_sail_CL = 0.0018311;
		double p0_sail_CD = -0.520349;
		double p1_sail_CD = 0.120098;
		double p2_sail_CD = -0.0131698;
		double p3_sail_CD = 0.00051913;
		double p0_kr_CL = -1.82637;
		double p1_kr_CL = 0.430472;
		double p2_kr_CL = -0.0453327;
		double p3_kr_CL = 0.00181221;
		double p0_kr_CD = -0.907082;
		double p1_kr_CD = 0.106607;
		double p2_kr_CD = -0.00416986;
		double p0_frk = 0.0324864;
		double p1_frk = 1.13068;
	  	// VARIABLE
	  	double simulation_time = 0;
	  	double v_ahx;
	  	double v_ahy;
	  	double v_ah;
	  	double alpha_ah;
	  	double v_aku;
	  	double v_akv;
	  	double v_ak;
	  	double alpha_ak;
	  	double corrected_keel_C_D;
	  	double L_k;
	  	double D_k;
	  	double alpha_tw;
	  	double v_awu;
	  	double v_awv;
	  	double v_aw;
	  	double alpha_aw;
	  	double alpha_s;
	  	double corrected_sail_C_D;
	  	double L_s;
	  	double D_s;
	  	double v_aru;
	  	double v_arv;
	  	double v_ar;
	  	double alpha_ar;
	  	double alpha_r;
	  	double corrected_rudder_C_D;
	  	double L_r;
	  	double D_r;
	  	Matrix4d C;
	  	Matrix4d J(){
	  		return Matrix4d {	{cos(xi),	-sin(xi)*cos(phi), 	0, 	0},
								{sin(xi),	cos(xi)*cos(phi), 	0, 	0},
								{0,		    0, 		            1,  0},
								{0,		    0, 		            0,  cos(phi)}};
	  	}
		Matrix4d M_RB{	{m, 			0, 		0, 		0},
						{0, 			m, 	    0, 		0},
						{0, 			0, 	    I_xx, 	0},
						{0, 			0, 	    0, 	    I_zz}};

		Matrix4d M_A{	{-x_u_dot, 		0, 			0, 				0},
						{0, 			-y_v_dot, 	0, 				0},
						{0, 			0, 	        -k_p_dot, 		0},
						{0, 			0, 	        0, 				-n_r_dot}};
        Matrix4d inverse_M = (M_RB+M_A).inverse();

		Matrix4d C_RB(){
			return Matrix4d {	{0, 		-m*r, 	0,	0},
								{m*r, 	    0, 		0,	0},
								{0, 		0, 		0,	0},
								{0, 		0, 		0,	0}};
		}

		Matrix4d C_A(){
			return Matrix4d {	{0, 		        0, 		0,  y_v_dot*v},
								{0, 		        0, 		0,	-x_u_dot*u},
								{0, 		        0, 		0,	0},
								{-y_v_dot*v, 	x_u_dot*u,  0,	0}};
		}
		Vector4d D_hell_yaw(){
		    return Vector4d(0, 0, c*phi_dot*abs(phi_dot), d*xi_dot*abs(xi_dot)*cos(phi));
		}
		Vector4d D_h(){
            v_ahx = -u+r*y_h;
            v_ahy = (-v-r*x_h+p*z_h)/cos(phi);
            v_ah = sqrt(v_ahx*v_ahx + v_ahy*v_ahy);
            alpha_ah = atan2(v_ahy, -v_ahx);
            return Vector4d(F_rh(v_ah)*cos(alpha_ah),
                            -F_rh(v_ah)*sin(alpha_ah)*cos(phi),
                            (-F_rh(v_ah)*sin(alpha_ah)*cos(phi))*abs(z_h),
                            F_rh(v_ah)*sin(alpha_ah)*cos(phi)*abs(x_h));
		}
		Vector4d D_keel(){// called D_k vector in the paper
		    v_aku = -u+r*y_k;
		    v_akv = -v-r*x_k+p*z_k;
		    v_ak = sqrt(v_aku*v_aku + v_akv*v_akv);
		    alpha_ak = atan2(v_akv, -v_aku);
		    L_k = 0.5*rho_water*A_k*v_ak*v_ak*keel_C_L(alpha_ak);
		    D_k = 0.5*rho_water*A_k*v_ak*v_ak*keel_C_D(alpha_ak);
		    return Vector4d(-L_k*sin(alpha_ak)+D_k*cos(alpha_ak),
		                    -L_k*cos(alpha_ak)-D_k*sin(alpha_ak),
		                    (-L_k*cos(alpha_ak)-D_k*sin(alpha_ak))*abs(z_k),
		                    (L_k*cos(alpha_ak)+D_k*sin(alpha_ak))*abs(x_k));
		}
		Vector4d D(){
			return D_keel()+D_h()+D_hell_yaw();
		}
		Vector4d g(){
            return Vector4d(0, 0, a*phi*phi+b*phi, 0);
		}
		Vector4d tau_s(double delta_s){
            v_awu = WM.get_v_tw()*cos(WM.get_alpha_tw()-xi)-u+r*y_s;
            v_awv = WM.get_v_tw()*sin(WM.get_alpha_tw()-xi)*cos(phi)-v-r*x_s+p*z_s;
            v_aw = sqrt(v_awu*v_awu + v_awv*v_awv);
            alpha_aw = atan2(v_awv, -v_awu);
            alpha_s = alpha_aw - delta_s;
            L_s = 0.5*rho_air*A_s*v_aw*v_aw*sail_C_L(alpha_s);
            D_s = 0.5*rho_air*A_s*v_aw*v_aw*sail_C_D(alpha_s);
            return Vector4d(L_s*sin(alpha_aw)-D_s*cos(alpha_aw),
                            L_s*cos(alpha_aw)+D_s*sin(alpha_aw),
                            (L_s*cos(alpha_aw)+D_s*sin(alpha_aw))*abs(z_s),
                            -(L_s*sin(alpha_aw)-D_s*cos(alpha_aw))*x_sm*sin(delta_s)+(L_s*cos(alpha_aw)+D_s*sin(alpha_aw))*(x_m-x_sm*cos(delta_s)));
		}
		Vector4d tau_r(double delta_r){
            v_aru = -u+r*y_r;
            v_arv = -v-r*x_r+p*z_r;
            v_ar = sqrt(v_aru*v_aru + v_arv*v_arv);
            alpha_ar = atan2(v_arv, -v_aru);
            alpha_r = alpha_ar-delta_r;
            L_r = 0.5*rho_water*A_r*v_ar*v_ar*rudder_C_L(alpha_r);
            D_r = 0.5*rho_water*A_r*v_ar*v_ar*rudder_C_D(alpha_r);
            // std::cout << "alpha_ar = " << alpha_ar*180./M_PI << std::endl;
            return Vector4d(-D_r, L_r, L_r*abs(z_r), -L_r*abs(x_r));
            /*return Vector4d(L_r*sin(alpha_ar)-D_r*cos(alpha_ar),
                            L_r*cos(alpha_ar)+D_r*sin(alpha_ar),
                            (L_r*cos(alpha_ar)+D_r*sin(alpha_ar))*abs(z_r),
                            -(L_r*cos(alpha_ar)+D_r*sin(alpha_ar))*abs(x_r));*/
		}
		Vector4d tau(double delta_s, double delta_r){
		    return tau_s(delta_s)+tau_r(delta_r);
		}
		double sail_C_L(double alpha){
			while (alpha>M_PI){
				alpha -= 2*M_PI;
			}
			while (alpha<-M_PI){
				alpha += 2*M_PI;
			}
			if (alpha>=0){
            	return p3_sail_CL*pow(abs(alpha), 9)+p2_sail_CL*pow(abs(alpha), 7)+p1_sail_CL*pow(abs(alpha), 5)+p0_sail_CL*pow(abs(alpha), 3)+(-pow(M_PI, 2)*p0_sail_CL-pow(M_PI, 4)*p1_sail_CL-pow(M_PI, 6)*p2_sail_CL-pow(M_PI, 8)*p3_sail_CL)*abs(alpha);
            }else{
            	return -(p3_sail_CL*pow(abs(alpha), 9)+p2_sail_CL*pow(abs(alpha), 7)+p1_sail_CL*pow(abs(alpha), 5)+p0_sail_CL*pow(abs(alpha), 3)+(-pow(M_PI, 2)*p0_sail_CL-pow(M_PI, 4)*p1_sail_CL-pow(M_PI, 6)*p2_sail_CL-pow(M_PI, 8)*p3_sail_CL)*abs(alpha));
            }
		}
		double sail_C_D(double alpha){
            while (alpha>M_PI){
				alpha -= 2*M_PI;
			}
			while (alpha<-M_PI){
				alpha += 2*M_PI;
			}
			alpha = abs(alpha);
			return p3_sail_CD*pow(alpha, 10)+p2_sail_CD*pow(alpha, 8)+p1_sail_CD*pow(alpha, 6)+p0_sail_CD*pow(alpha, 4)+((-0.1-4*p0_sail_CD*pow(M_PI, 3)-6*p1_sail_CD*pow(M_PI, 5)-8*p2_sail_CD*pow(M_PI, 7)-10*p3_sail_CD*pow(M_PI, 9))/(2*M_PI))*pow(alpha, 2)+0.1;
		}
		double rudder_C_L(double alpha){
            while (alpha>M_PI){
				alpha -= 2*M_PI;
			}
			while (alpha<-M_PI){
				alpha += 2*M_PI;
			}
			if (alpha>=0){
				alpha = abs(alpha);
            	return p3_kr_CL*pow(alpha, 9)+p2_kr_CL*pow(alpha, 7)+p1_kr_CL*pow(alpha, 5)+p0_kr_CL*pow(alpha, 3)+(-pow(M_PI, 2)*p0_kr_CL-pow(M_PI, 4)*p1_kr_CL-pow(M_PI, 6)*p2_kr_CL-pow(M_PI, 8)*p3_kr_CL)*alpha;
            }else{
            	alpha = abs(alpha);
            	return -(p3_kr_CL*pow(alpha, 9)+p2_kr_CL*pow(alpha, 7)+p1_kr_CL*pow(alpha, 5)+p0_kr_CL*pow(alpha, 3)+(-pow(M_PI, 2)*p0_kr_CL-pow(M_PI, 4)*p1_kr_CL-pow(M_PI, 6)*p2_kr_CL-pow(M_PI, 8)*p3_kr_CL)*alpha);
            }
		}
		double rudder_keel_C_D(double alpha){
		    while (alpha>M_PI){
				alpha -= 2*M_PI;
			}
			while (alpha<-M_PI){
				alpha += 2*M_PI;
			}
			alpha = abs(alpha);
            return p2_kr_CD*pow(alpha, 8)+p1_kr_CD*pow(alpha, 6)+p0_kr_CD*pow(alpha, 4)+((-p0_kr_CD*pow(M_PI, 4)-p1_kr_CD*pow(M_PI, 6)-p2_kr_CD*pow(M_PI, 8))/(pow(M_PI, 2)))*pow(alpha, 2);
		}
		double rudder_C_D(double alpha){
            return  rudder_keel_C_D(alpha) + pow(rudder_C_L(alpha), 2)/(M_PI*AR_r);
		}
		double keel_C_L(double alpha){
            return rudder_C_L(alpha);
		}
		double keel_C_D(double alpha){
            return  rudder_keel_C_D(alpha) + pow(keel_C_L(alpha), 2)/(M_PI*AR_k);
		}
		double F_rh(double v_ah){
			if (v_ah<0){
				std::cout << "Warning!!! During calculation of F_rh value (" << v_ah << ") out of bound [0;6]";
				v_ah = 0;
			}else if (v_ah>6){
				std::cout << "Warning!!! During calculation of F_rh value (" << v_ah << ") out of bound [0;6]";
				v_ah = 6;
			}
            return p0_frk*exp(p1_frk*v_ah)*pow(10, 3);
		}
		Wind_Model WM;
  	public:
	  	Sailing_Boat_Model(	Vector4d initial_eta = Vector4d {0, 0, 0, 0},
	  						Vector4d initial_nu = Vector4d {0, 0, 0, 0},
	  						Wind_Model given_WM = Wind_Model()){
	  		eta = initial_eta; // A deep copy is done: I have checked
	  		nu = initial_nu; // A deep copy is done: I have checked
	  		WM = given_WM;
	  		x = eta(0);
		    y = eta(1);
		    phi = eta(2);
		    xi = eta(3);
		    u = nu(0);
		    v = nu(1);
		    p = nu(2);
		    r = nu(3);
		    phi_dot = 0;
		    xi_dot = 0;
	  	}

        std::vector<double> hist_time, hist_x, hist_y, hist_phi, hist_xi, hist_u, hist_v, hist_p, hist_r,
                            hist_x_dot, hist_y_dot, hist_phi_dot, hist_xi_dot, hist_u_dot, hist_v_dot, hist_p_dot, hist_r_dot;
        Vector4d integration(double delta_s, double delta_r){
            C = C_RB()+C_A();
            eta_dot = J()*nu;
            nu_dot = -inverse_M*C*nu-inverse_M*D()-inverse_M*g()+inverse_M*tau(delta_s, delta_r);
            eta(0) = eta(0) + DT*eta_dot(0);
            eta(1) = eta(1) + DT*eta_dot(1);
            eta(2) = eta(2) + DT*eta_dot(2);
            eta(3) = eta(3) + DT*eta_dot(3);
            nu(0) = nu(0) + DT*nu_dot(0);
            nu(1) = nu(1) + DT*nu_dot(1);
            nu(2) = nu(2) + DT*nu_dot(2);
            nu(3) = nu(3) + DT*nu_dot(3);
            x = eta(0);
            y = eta(1);
            phi = eta(2);
            xi = eta(3);
            u = nu(0);
            v = nu(1);
            p = nu(2);
            r = nu(3);
            phi_dot = eta_dot(2);
		    xi_dot = eta_dot(3);
		    //Fill Graphs
		    /*hist_time.push_back(simulation_time);
		    hist_x.push_back(x);
		    hist_y.push_back(y);
		    hist_phi.push_back(phi);
		    hist_xi.push_back(xi);
		    hist_u.push_back(u);
		    hist_v.push_back(v);
		    hist_p.push_back(p);
		    hist_r.push_back(r);
            hist_x_dot.push_back(eta_dot(0));
            hist_y_dot.push_back(eta_dot(1));
            hist_phi_dot.push_back(eta_dot(2));
            hist_xi_dot.push_back(eta_dot(3));
            hist_u_dot.push_back(nu_dot(0));
            hist_v_dot.push_back(nu_dot(1));
            hist_p_dot.push_back(nu_dot(2));
            hist_r_dot.push_back(nu_dot(3));
            simulation_time += DT;*/
            return nu_dot;
        }

		Vector4d get_eta(){
			return eta;
		}

		Vector4d get_nu(){
			return nu;
		}
		void coefficient_graph(){
            int n = 361;
            int m = 61;
            std::vector<double> x(n), CLs(n), CDs(n), CLr(n), CDr(n), CLk(n), CDk(n), FRK(m), z(m);
            Vector4d s, r;
            for(int i=-180; i<181; ++i) {
                x.at(i+180) = i;
                CLs.at(i+180) = sail_C_L(i*M_PI/180.);
                CDs.at(i+180) = sail_C_D(i*M_PI/180.);
                CLr.at(i+180) = rudder_C_L(i*M_PI/180.);
                CDr.at(i+180) = rudder_C_D(i*M_PI/180.);
                CLk.at(i+180) = keel_C_L(i*M_PI/180.);
                CDk.at(i+180) = keel_C_D(i*M_PI/180.);
            }
            for(int i=0; i<61; ++i) {
            	z.at(i) = i/10.;
            	FRK.at(i) = F_rh(i/10.);
            }

            // Set the size of output image to 1200x780 pixels
            // plt::figure_size(1200, 780);
            // Plot line from given x and y data. Color is selected automatically.
            plt::figure();
            plt::plot(x, CLs, {{"label", "CLs"}});
            plt::plot(x, CDs, {{"label", "CDs"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Coefficient_S.png");
            plt::figure();
            plt::plot(x, CLr, {{"label", "CLr"}});
            plt::plot(x, CDr, {{"label", "CDr"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Coefficient_R.png");
            plt::figure();
            plt::plot(x, CLk, {{"label", "CLk"}});
            plt::plot(x, CDk, {{"label", "CDk"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Coefficient_K.png");
            plt::figure();
            plt::plot(z, FRK, {{"label", "FRK"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Coefficient_FRK.png");
            plt::show();
		}

		void dynamic_graph(){
		//hist_time, hist_x, hist_y, hist_phi, hist_xi, hist_u, hist_v, hist_p, hist_r,
        //hist_x_dot, hist_y_dot, hist_phi_dot, hist_xi_dot, hist_u_dot, hist_v_dot, hist_p_dot, hist_r_dot;

		    plt::figure();
            plt::plot(hist_time, hist_x, {{"label", "hist_x"}});
            plt::plot(hist_time, hist_y, {{"label", "hist_y"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_xy.png");

            plt::figure();
            plt::plot(hist_time, hist_phi, {{"label", "hist_phi"}});
            plt::plot(hist_time, hist_xi, {{"label", "hist_xi"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_phixi.png");

            plt::figure();
            plt::plot(hist_time, hist_u, {{"label", "hist_u"}});
            plt::plot(hist_time, hist_v, {{"label", "hist_v"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_uv.png");

            plt::figure();
            plt::plot(hist_time, hist_p, {{"label", "hist_p"}});
            plt::plot(hist_time, hist_r, {{"label", "hist_r"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_pr.png");

            plt::figure();
            plt::plot(hist_time, hist_x_dot, {{"label", "hist_x_dot"}});
            plt::plot(hist_time, hist_y_dot, {{"label", "hist_y_dot"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_xy_dot.png");

            plt::figure();
            plt::plot(hist_time, hist_phi_dot, {{"label", "hist_phi_dot"}});
            plt::plot(hist_time, hist_xi_dot, {{"label", "hist_xi_dot"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_phixi_dot.png");

            plt::figure();
            plt::plot(hist_time, hist_u_dot, {{"label", "hist_u_dot"}});
            plt::plot(hist_time, hist_v_dot, {{"label", "hist_v_dot"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_uv_dot.png");

            plt::figure();
            plt::plot(hist_time, hist_p_dot, {{"label", "hist_p_dot"}});
            plt::plot(hist_time, hist_r_dot, {{"label", "hist_r_dot"}});
            plt::grid(true);
            plt::legend();
            plt::save("./Dynamic_pr_dot.png");


            plt::show();
		}
};

int main(){
    MySocketServer MSS = MySocketServer();
    Wind_Model WM = Wind_Model(10, 0);
    Sailing_Boat_Model SBM = Sailing_Boat_Model(Vector4d(0, 0, 0, 0), Vector4d(0, 0, 0, 0), WM);
    double ellapsed_time = 0;
    std::vector<double> sending_parameters(8);
    std::vector<double> recived_input(2);
    Vector4d my_eta;
    double alpha_tw;
    Vector4d acceleration;
    double sail_positon = 0;
    double rudder_positon = 0;
    while(true){
        acceleration = SBM.integration(sail_positon, rudder_positon);
        if ((int)(ellapsed_time/DT) % 1000 == 0){
            recived_input = MSS.recive_input();
            if (recived_input[0] != -1000){
                sail_positon = recived_input[0];
                rudder_positon = recived_input[1];
            }
        }
        if ((int)(ellapsed_time/DT) % 10000 == 0){
            std::cout << "acceleration = [" << acceleration(0) << "; " << acceleration(1) << "; " << acceleration(2) << "; " << acceleration(3) << "]" << std::endl;
            std::cout << "Time: " << ellapsed_time << " s " << std::endl;
            std::cout << "recived from the visualizer: " << sail_positon << ", " << rudder_positon << std::endl;
            if (ellapsed_time>10){
                //break;
            }
        }
        my_eta = SBM.get_eta();
        alpha_tw = WM.get_alpha_tw();
        sending_parameters[0] = my_eta(0);
        sending_parameters[1] = my_eta(1);
        sending_parameters[2] = my_eta(2);
        sending_parameters[3] = my_eta(3);
        sending_parameters[4] = sail_positon;
        sending_parameters[5] = alpha_tw;
        sending_parameters[6] = atan2(acceleration(1), acceleration(0));
        sending_parameters[7] = rudder_positon;
        MSS.send_data(sending_parameters);
        ellapsed_time += DT;
    }
    SBM.dynamic_graph();
}

