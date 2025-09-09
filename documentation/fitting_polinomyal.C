
void format_line(TAttLine* line,int col,int sty){
    line->SetLineWidth(2);
    line->SetLineColor(col);
    line->SetLineStyle(sty);
}

/*double poly_5(double* vars, double* pars){
	return pars[0]*vars[0]*vars[0]*vars[0]*vars[0]*vars[0] //ax^5
		+ pars[1]*vars[0]*vars[0]*vars[0]*vars[0] //bx^4
		+ pars[2]*vars[0]*vars[0]*vars[0] //cx^3
		+ pars[3]*vars[0]*vars[0] //dx^2
		+ pars[4]*vars[0] // ex
		+ pars[5]; //f
}*/

int fitting_polinomyal(){

	
	
	double CLS[] = {0., 0.32, 0.71, 1.15, 1.3, 1.1, 1.0, 0.83, 0.69, 0.5, 0.21, -0.05, -0.29, -0.5, -0.7, -0.93, -1.1, -1.1, 0};
	double CDS[] = {0.1, 0.15, 0.2, 0.35, 0.55, 0.71, 0.91, 1.05, 1.16, 1.24, 1.29, 1.36, 1.34, 1.28, 1.25, 1.13, 0.9, 0.66, 0.1};
	double CLRK[] = {0.0, 0.75, 1.1, 1.19, 1.14, 1.0, 0.83, 0.63, 0.34, 0.0, -0.31, -0.63, -0.81, -0.99, -1.1, -1.17, -1.1, -0.72, 0.0};
	double CDRK[] = {0.0, 0.19, 0.49, 0.95, 1.35, 1.65, 1.9, 2.0, 2.1, 2.0, 1.9, 1.66, 1.36, 1.01, 0.5, 0.17, 0.06, 0.0};
	
	
	int num_of_point = 19;
	double raw_data_CL[] = {0.0, 0.75, 1.1, 1.19, 1.14, 1.0, 0.83, 0.63, 0.34, 0.0, -0.31, -0.63, -0.81, -0.99, -1.1, -1.17, -1.1, -0.72, 0.0};
	double raw_data_CD[] = {0.0, 0.19, 0.49, 0.95, 1.35, 1.65, 1.9, 2.0, 2.1, 2.0, 1.9, 1.66, 1.36, 1.01, 0.5, 0.17, 0.06, 0.0};
	double binsize = TMath::Pi()/18;
	
    gStyle->SetOptTitle(0); gStyle->SetOptStat(0);
    gStyle->SetOptFit(1111); gStyle->SetStatBorderSize(0);
    gStyle->SetStatX(.89); gStyle->SetStatY(.89);
	
	std::string Pi = to_string(TMath::Pi());
	
	std::string string_func_CL = "[3]*x**9+[2]*x**7+[1]*x**5+[0]*x**3+(-"+Pi+"**2*[0]-"+Pi+"**4*[1]-"+Pi+"**6*[2]-"+Pi+"**8*[3])*x";
	/* SAIL: */ std::string string_func_CD = "[3]*x**10+[2]*x**8+[1]*x**6+[0]*x**4+((-0.1-4*[0]*"+Pi+"**3-6*[1]*"+Pi+"**5-8*[2]*"+Pi+"**7-10*[3]*"+Pi+"**9)/(2*"+Pi+"))*x**2+0.1";
	/* Kell/Rudder: */std::string string_func_CD = "[2]*x**8+[1]*x**6+[0]*x**4+((-[0]*"+Pi+"**4-[1]*"+Pi+"**6-[2]*"+Pi+"**8)/("+Pi+"**2))*x**2";
	
	std::cout << "Cl: " << string_func_CL << std::endl;
	std::cout << "Cd: " << string_func_CD << std::endl;
	
	TF1* my_poly_function_CL = new TF1("poly_CL", string_func_CL.c_str(), 0, TMath::Pi());
	format_line(my_poly_function_CL, kRed, 2);
	TF1* my_poly_function_CD = new TF1("poly_CD", string_func_CD.c_str(), 0, TMath::Pi());
	format_line(my_poly_function_CL, kBlue, 2);
	
	my_poly_function_CL->SetParameter(0,1);
	my_poly_function_CL->SetParameter(1,1);
	my_poly_function_CL->SetParameter(2,1);
	my_poly_function_CL->SetParameter(3,1);
	my_poly_function_CL->SetParameter(4,1);
	my_poly_function_CL->SetParameter(5,1);
	
	TH1F* histo_CL = new TH1F("histo","Data;X vals;Y Vals", 19, 0-binsize/2, TMath::Pi()+binsize/2);
	TH1F* histo_CD = new TH1F("histo","Data;X vals;Y Vals", 19, 0-binsize/2, TMath::Pi()+binsize/2);
	
    for (int i=1; i<20; i++){
    	histo_CL->AddBinContent(i, raw_data_CL[i-1]);
    	histo_CD->AddBinContent(i, raw_data_CD[i-1]);
    }

	
	auto fit_CL = histo_CL->Fit(my_poly_function_CL, "S");
    fit_CL->Print();
    auto fit_CD = histo_CD->Fit(my_poly_function_CD, "S");
    fit_CD->Print();
    //TMatrixDSym covMatrix (fit_CL->GetCovarianceMatrix());
	//covMatrix.Print();
	
	my_poly_function_CL->Draw();
	histo_CL->Draw("Same");
	my_poly_function_CD->Draw("Same");
	histo_CD->Draw("Same");
	
    return 0;
}

//RESULT
/*
CLS
Minimizer is Minuit / Migrad
Chi2                      =     0.106973
NDf                       =           13
Edm                       =  6.11754e-10
NCalls                    =          169
p0                        =     -1.44183   +/-   0.116567    
p1                        =      0.35006   +/-   0.0136851   
p2                        =   -0.0410826   +/-   0.00122047  
p3                        =    0.0018311   +/-   9.87812e-05 
*/
/*
CDS
Minimizer is Minuit / Migrad
Chi2                      =     0.141968
NDf                       =           15
Edm                       =   1.5141e-13
NCalls                    =          174
p0                        =    -0.520349   +/-   0.0279502   
p1                        =     0.120098   +/-   0.00183377  
p2                        =   -0.0131698   +/-   0.000121884 
p3                        =   0.00051913   +/-   8.63418e-06 
*/
/*
CLRK
Minimizer is Minuit / Migrad
Chi2                      =     0.350148
NDf                       =           12
Edm                       =  3.88158e-14
NCalls                    =          142
p0                        =     -1.82637   +/-   1.18908     
p1                        =     0.430472   +/-   0.409538    
p2                        =   -0.0453327   +/-   0.0546256   
p3                        =   0.00181221   +/-   0.00245564  

*/
/*
CDRK
Minimizer is Minuit / Migrad
Chi2                      =     0.457345
NDf                       =           13
Edm                       =  5.26033e-17
NCalls                    =           82
p0                        =    -0.907082   +/-   0.415922    
p1                        =     0.106607   +/-   0.0646636   
p2                        =  -0.00416986   +/-   0.0031482  
*/
