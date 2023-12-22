#include <iostream> 
#include <fstream> 
#include <string>
 
int main() {
    std::ifstream file("lenet.raw", std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file" << std::endl;
        return 1;
    }
 
    float f;
    int i = 1;
    while (file.read(reinterpret_cast<char*>(&f), sizeof(f))) {      
        std::cout << "param " << i << ": " << f << std::endl;
        i++;
    }
 
    file.close();
    return 0;
}