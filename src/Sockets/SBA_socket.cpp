#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <Eigen/Dense>
#include <netdb.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

using Eigen::Vector4d;


#ifndef SBA_socket
#define SERVER_IP "127.0.0.1"
#define PORT    8080
#define MAXLINE 40
#define ip_server "localhost"

#define charsforvalues 20
#define maxnumofvalues 20
#define SBA_socket
#endif

void error(const char *msg){
	perror(msg);
	exit(1);
}

class MySocketServer{
    int serSockDes, readStatus;
    struct sockaddr_in serAddr, cliAddr;
    socklen_t cliAddrLen;
    char buff[1024] = {0};
    std::string msg_string = "Hello to you too!!!\n";
    const char * msg = msg_string.c_str();

    char msg_to_send[300];

    char temp_number[charsforvalues];
    int ii = 0;
    int num_recived_values = 0;

	public:
    MySocketServer(){
        if ((serSockDes = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            perror("socket creation error...\n");
            exit(-1);
        }

          //binding the port to ip and port
        serAddr.sin_family = AF_INET;
        serAddr.sin_port = htons(PORT);
        serAddr.sin_addr.s_addr = INADDR_ANY;

        if ((bind(serSockDes, (struct sockaddr*)&serAddr, sizeof(serAddr))) < 0) {
            perror("binding error...\n");
            close(serSockDes);
            exit(-1);
        }

        cliAddrLen = sizeof(cliAddr);
        readStatus = recvfrom(serSockDes, buff, 1024, 0, (struct sockaddr*)&cliAddr, &cliAddrLen);
        if (readStatus < 0) {
            perror("sending error...\n");
            close(serSockDes);
            exit(-1);
        }

        struct timeval read_timeout;
        read_timeout.tv_sec = 0;
        read_timeout.tv_usec = 100;
        setsockopt(serSockDes, SOL_SOCKET, SO_RCVTIMEO, &read_timeout, sizeof read_timeout);

        int rc;
        int optval;
        optval = 512;
        rc = setsockopt(serSockDes, SOL_SOCKET, SO_RCVBUF,(char *)&optval, sizeof(int)); // Setta al minimo il numero di elementi messi in coda!

        std::cout.write(buff, readStatus);
        std::cout << std::endl;

        if (sendto(serSockDes, msg, strlen(msg), 0, (struct sockaddr*)&cliAddr, cliAddrLen) < 0) {
            perror("sending error...\n");
            close(serSockDes);
            exit(-1);
        }
    }

    std::vector<double> input_from_visualizer = {0, 0};

    std::vector<double> recive_input(){
        readStatus = recvfrom(serSockDes, buff, 1024, 0, (struct sockaddr*)&cliAddr, &cliAddrLen);
        if (readStatus < 0) {
            input_from_visualizer[0] = -1000;
            input_from_visualizer[1] = -1000;
            return input_from_visualizer;
        }
        //std::cout.write(buff, readStatus);
        //std::cout << std::endl;
        ii=0;
        num_recived_values=0;
        for (int i=1; i<readStatus; i++){ // 1 because we don't care about parentesis
        	if (buff[i] != ',' && buff[i] != ']'){
        		temp_number[ii] = buff[i];
        		ii++;
        	}else{
        		for (int iii=ii; iii<charsforvalues; iii++){
        			temp_number[iii] = '0';
        		}
        		ii = 0;
        		input_from_visualizer[num_recived_values] = std::stod(temp_number);
        		//std::cout << recived_values[num_recived_values] << std::endl;
        		num_recived_values++;
        	}
        }
        return input_from_visualizer;
    }

	void send_data(std::vector<double> sp){
	    sprintf(msg_to_send, "[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]", sp[0], sp[1], sp[2], sp[3], sp[4], sp[5], sp[6], sp[7]);
 	    if (sendto(serSockDes, msg_to_send, strlen(msg_to_send), 0, (struct sockaddr*)&cliAddr, cliAddrLen) < 0) {
            perror("sending error...\n");
            close(serSockDes);
            exit(-1);
        }
        //std::cout.write(msg, strlen(msg));
        //std::cout << std::endl;
	}


    void close_connection(){
        close(serSockDes);
    }
};

class MySocketClient{
    int cliSockDes, readStatus;
    struct sockaddr_in serAddr;
    socklen_t serAddrLen;
    std::string msg_string = "Hello!!!\n";
    const char * msg = msg_string.c_str();
    char buff[1024] = {0};
	
	char temp_number[charsforvalues];
    int ii = 0;
    int num_recived_values = 0;
    std::vector<double> recived_values = std::vector<double>(maxnumofvalues);

    char msg_to_send[300];

    public:
    MySocketClient(){
    	for (int i=0; i<maxnumofvalues; i++){
    		recived_values[i] = 0.0;
    	}
        //create a socket
        if ((cliSockDes = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
            perror("socket creation error...\n");
            exit(-1);
        }

        //server socket address
        serAddr.sin_family = AF_INET;
        serAddr.sin_port = htons(PORT);
        serAddr.sin_addr.s_addr = inet_addr("127.0.0.1");

        int rc;
        int optval;
        optval = 512;
        rc = setsockopt(cliSockDes, SOL_SOCKET, SO_RCVBUF,(char *)&optval, sizeof(int)); // Setta al minimo il numero di elementi messi in coda!

        if (sendto(cliSockDes, msg, strlen(msg), 0, (struct sockaddr*)&serAddr, sizeof(serAddr)) < 0) {
            perror("sending error...\n");
            close(cliSockDes);
            exit(-1);
        }

        serAddrLen = sizeof(serAddr);
        readStatus = recvfrom(cliSockDes, buff, 1024, 0, (struct sockaddr*)&serAddr, &serAddrLen);
        if (readStatus < 0) {
            perror("reading error...\n");
            close(cliSockDes);
            exit(-1);
        }

        std::cout.write(buff, readStatus);
        std::cout << std::endl;
   	}

    int c = 0;

   	std::vector<double> reading(double lambda, double sigma){
   	    readStatus = recvfrom(cliSockDes, buff, 1024, 0, (struct sockaddr*)&serAddr, &serAddrLen);
        if (readStatus < 0) {
            perror("reading error...\n");
            close(cliSockDes);
            exit(-1);
        }
        // std::cout.write(buff, readStatus);
        // std::cout << std::endl;
        ii=0;
        num_recived_values=0;
        for (int i=1; i<readStatus; i++){ // 1 because we don't care about parentesis
        	if (buff[i] != ',' && buff[i] != ']'){
        		temp_number[ii] = buff[i];
        		ii++;
        	}else{
        		for (int iii=ii; iii<charsforvalues; iii++){
        			temp_number[iii] = '0';
        		}
        		ii = 0;
        		recived_values[num_recived_values] = std::stod(temp_number);
        		//std::cout << recived_values[num_recived_values] << std::endl;
        		num_recived_values++;
        	}
        }

        sprintf(msg_to_send, "[%.6f,%.6f]", lambda, sigma);
 	    if (sendto(cliSockDes, msg_to_send, strlen(msg_to_send), 0, (struct sockaddr*)&serAddr, sizeof(serAddr)) < 0) {
            perror("sending error...\n");
            close(cliSockDes);
            exit(-1);
        }

        //std::cout <<"[" << c << "] Sended: " << msg_to_send << std::endl;
        c++;
        return recived_values;
   	}

   	void close_connection(){
    	close(cliSockDes);
    }
};

