#include <fstream>
#include <sstream>
#include <glm/glm.hpp>
#include "ImportedModel.h"
using namespace std;

ImportedModel::ImportedModel(string filePath){
    ModelImporter modelImporter = ModelImporter();
    modelImporter.parseOBJ(filePath);
    numVertices = modelImporter.getNumVertices();
    std::cout << "Apro " << filePath << " fil con " << numVertices << " vertexs!" << std::endl;
    std::vector<float> verts = modelImporter.getVertices();
    std::vector<float> tcs = modelImporter.getTextCoords();
    std::vector<float> normals = modelImporter.getNormalVecs();

    for (int i=0; i<numVertices; i++){
        vertices.push_back(glm::vec3(verts[i*3], verts[i*3+1], verts[i*3+2]));
        textCoords.push_back(glm::vec2(tcs[i*2], tcs[i*2+1]));
        normalVecs.push_back(glm::vec3(normals[i*3], normals[i*3+1], normals[i*3+2]));
    }
}

int ImportedModel::getNumVertices(){
    return numVertices;
}
std::vector<glm::vec3> ImportedModel::getVertices(){
    return vertices;
}
std::vector<glm::vec2> ImportedModel::getTextCoords(){
    return textCoords;
}
std::vector<glm::vec3> ImportedModel::getNormalVecs(){
    return normalVecs;
}

ModelImporter::ModelImporter(){}

void ModelImporter::parseOBJ(string filePath){
    float x, y, z;
    string content;
    ifstream fileStream(filePath.c_str(), ios::in);
    string line;
    //cout<<"FILE:"<<"\n";
    int raws_done = 0;
    try{
        while (!fileStream.eof()){
        	//std::cout << "Raws Done: " << raws_done << std::endl;
            raws_done++;
            getline(fileStream, line);
            //cout<<line<<"\n";
            if (line.compare(0,2,"v ") == 0){
                stringstream ss(line.erase(0,1));
                ss >> x;
                ss >> y;
                ss >> z;
                vertVals.push_back(x);
                vertVals.push_back(y);
                vertVals.push_back(z);
            }
            if (line.compare(0,2,"vt") == 0){
                stringstream ss(line.erase(0,2));
                ss >> x;
                ss >> y;
                stVals.push_back(x);
                stVals.push_back(y);
                //cout<<"vt[?]="<<x<<"\n";
            }
            if (line.compare(0,2,"vn") == 0){
                stringstream ss(line.erase(0,2));
                ss >> x;
                ss >> y;
                ss >> z;
                normVals.push_back(x);
                normVals.push_back(y);
                normVals.push_back(z);
            }
            if (line.compare(0,2,"f ") == 0){
                string oneCorner, v, t, n;
                stringstream ss(line.erase(0,2));
                for (int i=0; i<3; i++){
                    getline(ss, oneCorner, ' ');
                    stringstream oneCornerSS(oneCorner);
                    //cout<<oneCornerSS.str()<<"\n";

                    getline(oneCornerSS, v, '/');
                    getline(oneCornerSS, t, '/');

                    getline(oneCornerSS, n, '/');
                    //cout<<"["<<v<<", "<<t<<", "<<n<<"]"<<"\n";

                    int vertRef = (stoi(v)-1)*3;
                    int tcRef;
                    if (t != ""){
                        tcRef = (stoi(t)-1)*2;
                    }

                    int normRef = (stoi(n)-1)*3;
                    //cout<<"["<<vertRef<<", "<<tcRef<<", "<<normRef<<"]"<<"\n";

                    triangleVerts.push_back(vertVals[vertRef]);
                    triangleVerts.push_back(vertVals[vertRef+1]);
                    triangleVerts.push_back(vertVals[vertRef+2]);
                    //cout<<"triangleVerts DONE!"<<"\n";

                    if (t != ""){
                        textureCoords.push_back(stVals[tcRef]);
                        textureCoords.push_back(stVals[tcRef+1]);
                    }else{
                        stVals.push_back(0);
                        stVals.push_back(0);
                        textureCoords.push_back(0);
                        textureCoords.push_back(0);
                    }
                    //cout<<"textureCoords DONE!"<<"\n";

                    normals.push_back(normVals[normRef]);
                    normals.push_back(normVals[normRef+1]);
                    normals.push_back(normVals[normRef+2]);
                    //cout<<"triangleVerts DONE!"<<"\n";
                }
            }
        }
    }catch (...) {
        std::cout << "Raws Done: " << raws_done << std::endl;
    }
    /*cout<<"END FILE\n";
    cout<<"Vertex:\n";
    for (int i=0; i<vertVals.size(); i++){
        cout<<"V["<<i<<"]="<<vertVals[i]<<", \n";
    }
    cout<<"Faces:\n";
    for (int i=0; i<triangleVerts.size(); i++){
        cout<<"TV["<<i<<"]="<<triangleVerts[i]<<", \n";
    }*/
}

int ModelImporter::getNumVertices() {
    return (triangleVerts.size()/3);
}
std::vector<float> ModelImporter::getVertices(){
    return triangleVerts;
}
std::vector<float> ModelImporter::getTextCoords(){
    return textureCoords;
}
std::vector<float> ModelImporter::getNormalVecs(){
    return normals;
}
