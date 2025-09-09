#include <vector>
#include <glm/glm.hpp>

class Cube{
    protected:
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normalVecs;
    std::vector<glm::vec2> texture;
    std::vector<glm::vec3> single_vertices;
    int num_of_vertices = 0;
    int num_of_single_vertices = 0;
    double min_x, max_x, min_y, max_y;
    public:
    Cube(double side_size, int points_for_side){
        double trg_side = side_size/(1.0*points_for_side);
        double x, y, z;
        x = side_size/2;
        for (int iy = 0; iy < points_for_side; iy++){
            for (int iz = 0; iz < points_for_side; iz++){
                y = (side_size*(1.0*iy)/(1.0*points_for_side))-side_size/2;
                z = (side_size*(1.0*iz)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y+trg_side,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                normalVecs.push_back(glm::vec3(1,0,0));
                normalVecs.push_back(glm::vec3(1,0,0));
                normalVecs.push_back(glm::vec3(1,0,0));
                normalVecs.push_back(glm::vec3(1,0,0));
                normalVecs.push_back(glm::vec3(1,0,0));
                normalVecs.push_back(glm::vec3(1,0,0));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z));
                single_vertices.push_back(glm::vec3(x,y,z+trg_side));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z+trg_side));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(y, z));
                texture.push_back(glm::vec2(y+trg_side,z));
                texture.push_back(glm::vec2(y,z+trg_side));
                texture.push_back(glm::vec2(y+trg_side,z+trg_side));
                texture.push_back(glm::vec2(y,z+trg_side));
                texture.push_back(glm::vec2(y+trg_side,z));
            }
        }
        x = -side_size/2;
        for (int iy = 0; iy < points_for_side; iy++){
            for (int iz = 0; iz < points_for_side; iz++){
                y = (side_size*(1.0*iy)/(1.0*points_for_side))-side_size/2;
                z = (side_size*(1.0*iz)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                normalVecs.push_back(glm::vec3(-1,0,0));
                normalVecs.push_back(glm::vec3(-1,0,0));
                normalVecs.push_back(glm::vec3(-1,0,0));
                normalVecs.push_back(glm::vec3(-1,0,0));
                normalVecs.push_back(glm::vec3(-1,0,0));
                normalVecs.push_back(glm::vec3(-1,0,0));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z));
                single_vertices.push_back(glm::vec3(x,y,z+trg_side));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z+trg_side));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
            }
        }
        z = side_size/2;
        for (int ix = 0; ix < points_for_side; ix++){
            for (int iy = 0; iy < points_for_side; iy++){
                x = (side_size*(1.0*ix)/(1.0*points_for_side))-side_size/2;
                y = (side_size*(1.0*iy)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y+trg_side,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y+trg_side, z));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
            }
        }
        z = -side_size/2;
        for (int ix = 0; ix < points_for_side; ix++){
            for (int iy = 0; iy < points_for_side; iy++){
                x = (side_size*(1.0*ix)/(1.0*points_for_side))-side_size/2;
                y = (side_size*(1.0*iy)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y+trg_side,z));
                vertices.push_back(glm::vec3(x,y+trg_side,z));
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y+trg_side,z));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y+trg_side,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y+trg_side, z));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
            }
        }
        y = side_size/2;
        for (int ix = 0; ix < points_for_side; ix++){
            for (int iz = 0; iz < points_for_side; iz++){
                x = (side_size*(1.0*ix)/(1.0*points_for_side))-side_size/2;
                z = (side_size*(1.0*iz)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z+trg_side));
                vertices.push_back(glm::vec3(x+trg_side,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                normalVecs.push_back(glm::vec3(0,0,1));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y,z+trg_side));
                single_vertices.push_back(glm::vec3(x+trg_side,y,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y, z+trg_side));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
            }
        }
        y = -side_size/2;
        for (int ix = 0; ix < points_for_side; ix++){
            for (int iz = 0; iz < points_for_side; iz++){
                x = (side_size*(1.0*ix)/(1.0*points_for_side))-side_size/2;
                z = (side_size*(1.0*iz)/(1.0*points_for_side))-side_size/2;
                vertices.push_back(glm::vec3(x,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z));
                vertices.push_back(glm::vec3(x+trg_side,y,z+trg_side));
                vertices.push_back(glm::vec3(x+trg_side,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z+trg_side));
                vertices.push_back(glm::vec3(x,y,z));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                normalVecs.push_back(glm::vec3(0,0,-1));
                num_of_vertices += 6;

                single_vertices.push_back(glm::vec3(x,y,z));
                single_vertices.push_back(glm::vec3(x,y,z+trg_side));
                single_vertices.push_back(glm::vec3(x+trg_side,y,z));
                single_vertices.push_back(glm::vec3(x+trg_side,y, z+trg_side));
                num_of_single_vertices += 4;
                
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
                texture.push_back(glm::vec2(1.0,1.0));
            }
        }
        min_x = -1;
        max_x = 1;
        min_y = -1;
        max_y = 1;
        normalize();
    }
    void normalize(){
    	double new_min_x = 1;
    	double new_max_x = 0;
    	double new_min_y = 1;
    	double new_max_y = 0;
    	for (int i=0; i<num_of_vertices; i++){
    		texture[i].x = (texture[i].x-min_x)/(max_x-min_x);
            texture[i].y = (texture[i].y-min_y)/(max_y-min_y);
            if (texture[i].x>new_max_x){
            	new_max_x=texture[i].x;
            }else if (texture[i].x<new_min_x){
            	new_min_x=texture[i].x;
            }
            
            if (texture[i].y>new_max_y){
            	new_max_y=texture[i].y;
            }else if (texture[i].y<new_min_y){
            	new_min_y=texture[i].y;
            }
    	}
    	std::cout << "Before: [" << min_x << "; " << max_x << "] [" << min_y << "; "  << max_y << "]" << std::endl;
    	std::cout << "After: [" << new_min_x << "; " << new_max_x << "] [" << new_min_y << "; "  << new_max_y << "]" << std::endl;
    }
    std::vector<glm::vec3> getVertices(){
        return vertices;
    }
    std::vector<glm::vec3> getNormalVecs(){
        return normalVecs;
    }
    std::vector<glm::vec2> getTexture(){
        return texture;
    }
    int getNumVertices(){
        return num_of_vertices;
    }
    int getNumSingleVertices(){
        return num_of_single_vertices;
    }
    glm::vec3 get_index(int i){
        return single_vertices[i];
    }
};

class Sphere: public Cube{
	double y_lim = 2;
    public:
    Sphere(double side_size, int points_for_side):Cube(side_size, points_for_side){
    	double norm;
        double x2, y2, z2;
        double lambda, phi;
        double map_x, map_y;
        for (int i=0; i<num_of_vertices; i++){
            x2 = vertices[i].x*vertices[i].x;
            y2 = vertices[i].y*vertices[i].y;
            z2 = vertices[i].z*vertices[i].z;
            
            vertices[i].x = vertices[i].x*sqrt(1-y2/2-z2/2+y2*z2/3);
            vertices[i].y = vertices[i].y*sqrt(1-x2/2-z2/2+x2*z2/3);
            vertices[i].z = vertices[i].z*sqrt(1-y2/2-x2/2+y2*x2/3);
            
            norm = sqrt(vertices[i].x*vertices[i].x+vertices[i].y*vertices[i].y+vertices[i].z*vertices[i].z);
            
            vertices[i].x = vertices[i].x/norm;
            vertices[i].y = vertices[i].y/norm;
            vertices[i].z = vertices[i].z/norm;
            
            phi = asin(vertices[i].y);
			lambda = atan2(vertices[i].x, -vertices[i].z);
            
            //map_x = lambda;
            //map_y = log(tan(M_PI/4 + phi/2));
            
            map_x = lambda;
            map_y = phi;
            
            if (map_y > y_lim){
            	map_y = y_lim;
            }else if (map_y < -y_lim){
            	map_y = -y_lim;
            }
            
            texture[i].x = map_x;
            texture[i].y = map_y;
            
            if (map_x>max_x){
            	max_x=map_x;
            }else if (map_x<min_x){
            	min_x=map_x;
            }
            
            if (map_y>max_y){
            	max_y=map_y;
            }else if (map_y<min_y){
            	min_y=map_y;
            }
            
            
            
            
            normalVecs[i].x = vertices[i].x;
            normalVecs[i].y = vertices[i].y;
            normalVecs[i].z = vertices[i].z;
            if (i<num_of_single_vertices){
                x2 = single_vertices[i].x*single_vertices[i].x;
            	y2 = single_vertices[i].y*single_vertices[i].y;
            	z2 = single_vertices[i].z*single_vertices[i].z;

                single_vertices[i].x = single_vertices[i].x*sqrt(1-y2/2-z2/2+y2*z2/3);
            	single_vertices[i].y = single_vertices[i].y*sqrt(1-x2/2-z2/2+x2*z2/3);
            	single_vertices[i].z = single_vertices[i].z*sqrt(1-y2/2-x2/2+y2*x2/3);
            	
            	norm = sqrt(single_vertices[i].x*single_vertices[i].x+single_vertices[i].y*single_vertices[i].y+single_vertices[i].z*single_vertices[i].z);
            
		        single_vertices[i].x = single_vertices[i].x/norm;
		        single_vertices[i].y = single_vertices[i].y/norm;
		        single_vertices[i].z = single_vertices[i].z/norm;
            }
        }
        normalize();
    }
};
