#include <vector>

class ImportedModel{
private:
    int numVertices;
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec2> textCoords;
    std::vector<glm::vec3> normalVecs;
public:
    ImportedModel(std::string);
    int getNumVertices();
    std::vector<glm::vec3> getVertices();
    std::vector<glm::vec2> getTextCoords();
    std::vector<glm::vec3> getNormalVecs();
};

class ModelImporter{
private:
    //values readed from .OBJ file
    std::vector<float> vertVals;
    std::vector<float> stVals;
    std::vector<float> normVals;
    
    //values used for data modification
    std::vector<float> triangleVerts;
    std::vector<float> textureCoords;
    std::vector<float> normals;
public:
    ModelImporter();
    void parseOBJ(std::string);
    int getNumVertices();
    std::vector<float> getVertices();
    std::vector<float> getTextCoords();
    std::vector<float> getNormalVecs();
};
