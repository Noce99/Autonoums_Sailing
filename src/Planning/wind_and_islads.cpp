#include <iostream>
#include <chrono>
#include <thread> //std::this_thread::sleep_for(std::chrono::milliseconds(50));
#include <string.h>
#include <vector>
#include <cmath>
#include <matplot/matplot.h>
#include <random>

using namespace matplot;

#define zone_of_interest 2

#define islands_x_y_size_mean 10
#define islands_x_y_size_variance 5
#define islands_x_y_height_mean 5
#define islands_x_y_height_variance 2

#define wind_x_y_size_mean 150
#define wind_x_y_size_variance 35
#define wind_x_y_height_mean 100
#define wind_x_y_height_variance 15

class gaussian{
    double A, sigma_x, sigma_y, x0, y0, theta;

public:

    gaussian(double x, double y, double x_size, double y_size, double my_theta, double height){
        A = height;
        x0 = x;
        y0 = y;
        sigma_x = x_size/2;
        sigma_y = y_size/2;
        theta = my_theta;
    }

    double at(double x, double y){
        return A*exp(-(x*cos(theta)-y*sin(theta)-x0)*(x*cos(theta)-y*sin(theta)-x0)/(2*sigma_x*sigma_x)-(x*sin(theta)+y*cos(theta)-y0)*(x*sin(theta)+y*cos(theta)-y0)/(2*sigma_y*sigma_y));
    }

    std::vector<double> gradient_at(double x, double y){
        return std::vector<double>{ -at(x,y)*(cos(theta)*(x*cos(theta)-y*sin(theta)-x0)/(sigma_x*sigma_x) + sin(theta)*(x*sin(theta)+y*cos(theta)-y0)/(sigma_y*sigma_y)),
                                    -at(x,y)*(-sin(theta)*(x*cos(theta)-y*sin(theta)-x0)/(sigma_x*sigma_x) + cos(theta)*(x*sin(theta)+y*cos(theta)-y0)/(sigma_y*sigma_y))};
    }

    std::vector<double> perpendicular_gradient_at(double x, double y){
        return std::vector<double>{ at(x,y)*(-sin(theta)*(x*cos(theta)-y*sin(theta)-x0)/(sigma_x*sigma_x) + cos(theta)*(x*sin(theta)+y*cos(theta)-y0)/(sigma_y*sigma_y)),
                                    -at(x,y)*(cos(theta)*(x*cos(theta)-y*sin(theta)-x0)/(sigma_x*sigma_x) + sin(theta)*(x*sin(theta)+y*cos(theta)-y0)/(sigma_y*sigma_y))};
    }


    void plot(int point_along_x){
        double resolution = 4*sigma_x/(1.0*point_along_x);
        double min_x = x0 - 2*sigma_x;
        double max_x = x0 + 2*sigma_x;
        double min_y = y0 - 2*sigma_y;
        double max_y = y0 + 2*sigma_y;
        double min, max;
        if (min_x<=min_y){
            min = min_x;
        }else{
            min = min_y;
        }
        if (max_x>=max_y){
            max = max_x;
        }else{
            max = max_y;
        }
        auto [X, Y] = meshgrid(iota(min, resolution, max));
        auto Z = transform(X, Y, [this](double x, double y){return at(x, y);});
        mesh(X, Y, Z);
        title("Prima Isolettiana!");
        show();
    }

    void plot_gradient(int point_along_x){
        double resolution = 4*sigma_x/(1.0*point_along_x);
        double min_x = x0 - 2*sigma_x;
        double max_x = x0 + 2*sigma_x;
        double min_y = y0 - 2*sigma_y;
        double max_y = y0 + 2*sigma_y;
        double min, max;
        if (min_x<=min_y){
            min = min_x;
        }else{
            min = min_y;
        }
        if (max_x>=max_y){
            max = max_x;
        }else{
            max = max_y;
        }
        auto cordinates = meshgrid(iota(min, resolution, max));
        auto X = std::get<0>(cordinates);
        auto Y = std::get<1>(cordinates);
        auto Z = transform(X, Y, [this](double x, double y){return at(x, y);});
        auto cont = contour(X, Y, Z);
        cont->n_levels(60);

        hold(on);

        cordinates = meshgrid(iota(min, resolution*10, max));
        X = std::get<0>(cordinates);
        Y = std::get<1>(cordinates);
        vector_2d u = transform(X, Y, [this](double x, double y) { return gradient_at(x, y)[0]; });
        vector_2d v = transform(X, Y, [this](double x, double y) { return gradient_at(x, y)[1]; });

        quiver(X, Y, u, v, 0.5);

        hold(on);

        u = transform(X, Y, [this](double x, double y) { return perpendicular_gradient_at(x, y)[0]; });
        v = transform(X, Y, [this](double x, double y) { return perpendicular_gradient_at(x, y)[1]; });

        quiver(X, Y, u, v, 0.5);

        show();
    }
};

class gaussian_accumulator{
    double x_map_size, y_map_size;
    std::vector<gaussian> all_gaussian;
    int gaussian_num;

    public:
    gaussian_accumulator(double my_x_map_size, double my_y_map_size, int my_gaussian_num, double x_y_size_mean, double x_y_size_variance, double x_y_height_mean, double x_y_height_variance){
        x_map_size = my_x_map_size;
        y_map_size = my_y_map_size;
        gaussian_num = my_gaussian_num;
        double x, y, x_size, y_size, theta, height;
        std::default_random_engine generator;
        std::normal_distribution<double> xy_size_gauss(x_y_size_mean, x_y_size_variance);
        std::normal_distribution<double> xy_height_gauss(x_y_height_mean, x_y_height_variance);
        for (int i=0; i<gaussian_num; i++){
            x = (rand()*1.0/RAND_MAX)*x_map_size - x_map_size/2;
            y = (rand()*1.0/RAND_MAX)*y_map_size - y_map_size/2;
            x_size = xy_size_gauss(generator);
            if (x_size<1){
                x_size = 1;
            }
            y_size = xy_size_gauss(generator);
            if (y_size<1){
                y_size = 1;
            }
            theta = (rand()*1.0/RAND_MAX)*2*M_PI;
            height = xy_height_gauss(generator);
            all_gaussian.push_back(gaussian(x, y, x_size, y_size, theta, height));
            std::cout << "Created a gaussian in [" << x << "; " << y << "] of size [" << x_size << "; " << y_size << "] of theta " << theta << " rad and height " << height << std::endl;
        }
    }

    double at(double x, double y){
        double result = 0;
        for (int i=0; i<gaussian_num; i++){
            result += all_gaussian[i].at(x, y);
        }
        return result;
    }
    std::vector<double> normalized_gradient_at(double x, double y){
        std::vector<double> result = {0, 0};
        std::vector<double> tmp;
        double norm;
        for (int i=0; i<gaussian_num; i++){
            tmp = all_gaussian[i].gradient_at(x, y);
            result[0] += tmp[0];
            result[1] += tmp[1];
        }
        return result;
    }

    std::vector<double> normalized_perpendicular_gradient_at(double x, double y){
        std::vector<double> result = {0, 0};
        std::vector<double> tmp;
        double norm;
        for (int i=0; i<gaussian_num; i++){
            tmp = all_gaussian[i].perpendicular_gradient_at(x, y);
            result[0] += tmp[0];
            result[1] += tmp[1];
        }
        return result;
    }

    void plot(int point_along_x){
        double resolution = x_map_size/(1.0*point_along_x);
        double min, max;
        if (x_map_size>=y_map_size){
            min = -x_map_size/2;
            max = x_map_size/2;
        }else{
            min = -y_map_size/2;
            max = y_map_size/2;
        }
        auto [X, Y] = meshgrid(iota(min, resolution, max));
        auto Z = transform(X, Y, [this](double x, double y){return at(x, y);});
        auto my_meshc = meshc(X, Y, Z);
        my_meshc->contour_levels(200);
        title("Some Paraboloids");
        zlim({xlim()[0], xlim()[1]});

        show();
    }

    void plot_gradient(int point_along_x){
        double resolution = x_map_size/(1.0*point_along_x);
        double min, max;
        if (x_map_size>=y_map_size){
            min = -x_map_size/2;
            max = x_map_size/2;
        }else{
            min = -y_map_size/2;
            max = y_map_size/2;
        }
        auto cordinates = meshgrid(iota(min, resolution, max));
        auto X = std::get<0>(cordinates);
        auto Y = std::get<1>(cordinates);
        auto Z = transform(X, Y, [this](double x, double y){return at(x, y)-250;});
        auto cont = contour(X, Y, Z);
        cont->n_levels(60);

        hold(on);

        cordinates = meshgrid(iota(min, resolution*5, max));
        X = std::get<0>(cordinates);
        Y = std::get<1>(cordinates);
        vector_2d u = transform(X, Y, [this](double x, double y) { return normalized_gradient_at(x, y)[0]; });
        vector_2d v = transform(X, Y, [this](double x, double y) { return normalized_gradient_at(x, y)[1]; });

        title("Some Paraboloids Gradients");
        quiver(X, Y, u, v, 0.5);

        hold(on);

        u = transform(X, Y, [this](double x, double y) { return normalized_perpendicular_gradient_at(x, y)[0]; });
        v = transform(X, Y, [this](double x, double y) { return normalized_perpendicular_gradient_at(x, y)[1]; });

        title("Some Paraboloids Gradients");
        quiver(X, Y, u, v, 0.5);
        show();
    }

    void plot_both(int point_along_x){
        double resolution = x_map_size/(1.0*point_along_x);
        double min, max;
        if (x_map_size>=y_map_size){
            min = -x_map_size/2;
            max = x_map_size/2;
        }else{
            min = -y_map_size/2;
            max = y_map_size/2;
        }
        auto cordinates = meshgrid(iota(min, resolution, max));
        auto X = std::get<0>(cordinates);
        auto Y = std::get<1>(cordinates);
        auto Z = transform(X, Y, [this](double x, double y){return at(x, y)-250;});
        auto my_meshc = meshc(X, Y, Z);
        my_meshc->contour_levels(40);
        title("Some Gaussian");
        zlim({xlim()[0], xlim()[1]});

        hold(on);

        cordinates = meshgrid(iota(min, resolution*10, max));
        X = std::get<0>(cordinates);
        Y = std::get<1>(cordinates);
        vector_2d u = transform(X, Y, [this](double x, double y) { return normalized_gradient_at(x, y)[0]; });
        vector_2d v = transform(X, Y, [this](double x, double y) { return normalized_gradient_at(x, y)[1]; });

        title("Some Gaussian Gradients");
        quiver(X, Y, u, v, 0.5);

        show();
    }
};

int main(){
    srand (time(NULL));
    //paraboloid_accumulator wind = paraboloid_accumulator(500, 500, 2, wind_x_y_size_mean, wind_x_y_size_variance, wind_x_y_height_mean, wind_x_y_height_variance);
    //wind.plot_gradient(30);
    //wind.plot(100);
    //gaussian_accumulator arcipelago = gaussian_accumulator(500, 500, 30, islands_x_y_size_mean, islands_x_y_size_variance, islands_x_y_height_mean, islands_x_y_height_variance);
    gaussian_accumulator wind = gaussian_accumulator(500, 500, 3, wind_x_y_size_mean, wind_x_y_size_variance, wind_x_y_height_mean, wind_x_y_height_variance);
    wind.plot_gradient(100);
}

