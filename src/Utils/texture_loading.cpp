#include <SOIL2.h>
#include <stdio.h>
#include <stdlib.h> 
#include <unistd.h>
#include <sys/wait.h>

bool LOADING = true;
int loading_texture_pipe[2];

std::vector<string> preloded_texture;

void preLoadTexture(string texImagePath){
	preloded_texture.push_back(texImagePath);
}

std::vector<unsigned int> loadTextures(){
	std::vector<unsigned int> ids;
	for (int i=0; i<preloded_texture.size(); i++){
		ids.push_back(SOIL_load_OGL_texture(preloded_texture[i].c_str(),
										SOIL_LOAD_AUTO,
										SOIL_CREATE_NEW_ID,
										SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT));
	}
	return ids;
}

void test_fork(){
	printf("hello world (pid:%d)\n", (int) getpid());
	int pID = fork();
	if (pID < 0) {
		printf("fork failed\n");
		exit(1);
	} else if (pID == 0) {
		printf("hello, I am child (pid:%d)\n", (int) getpid());
		for (int i=0; i<100000000; i++){}
		exit(0);
	} else {
		printf("hello, I am parent of %d (pid:%d)\n", pID, (int) getpid());
	}
	while (true) {
		int status;
		pid_t done = wait(&status);
		if (done == -1) {
		    if (errno == ECHILD) break; // no more child processes
		}
	}
}

int test_pipe(){
	// We use two pipes
	// First pipe to send input string from parent
	// Second pipe to send concatenated string from child

	int fd1[2]; // Used to store two ends of first pipe
	int fd2[2]; // Used to store two ends of second pipe

	char fixed_str[] = "forgeeks.org";
	char input_str[100];
	pid_t p;

	if (pipe(fd1) == -1) {
		fprintf(stderr, "Pipe Failed");
		return 1;
	}
	if (pipe(fd2) == -1) {
		fprintf(stderr, "Pipe Failed");
		return 1;
	}

	scanf("%s", input_str);
	p = fork();

	if (p < 0) {
		fprintf(stderr, "fork Failed");
		return 1;
	}

	// Parent process
	else if (p > 0) {
		char concat_str[100];

		close(fd1[0]); // Close reading end of first pipe

		// Write input string and close writing end of first
		// pipe.
		write(fd1[1], input_str, strlen(input_str) + 1);
		close(fd1[1]);

		// Wait for child to send a string
		wait(NULL);

		close(fd2[1]); // Close writing end of second pipe

		// Read string from child, print it and close
		// reading end.
		read(fd2[0], concat_str, 100);
		printf("Concatenated string %s\n", concat_str);
		close(fd2[0]);
	}

	// child process
	else {
		close(fd1[1]); // Close writing end of first pipe

		// Read a string using first pipe
		char concat_str[100];
		read(fd1[0], concat_str, 100);

		// Concatenate a fixed string with it
		int k = strlen(concat_str);
		int i;
		for (i = 0; i < strlen(fixed_str); i++)
			concat_str[k++] = fixed_str[i];

		concat_str[k] = '\0'; // string ends with '\0'

		// Close both reading ends
		close(fd1[0]);
		close(fd2[0]);

		// Write concatenated string and close writing end
		write(fd2[1], concat_str, strlen(concat_str) + 1);
		close(fd2[1]);

		exit(0);
	}
}

int my_fork_pipe_test(){
	int fd1[2];
	if (pipe(fd1) == -1) {
		fprintf(stderr, "Pipe Failed");
		return 1;
	}
	int pID = fork();
	if (pID < 0) {
		printf("fork failed\n");
		exit(1);
	} else if (pID == 0) {
		char message[] = "I'm a message from child!";
		printf("hello, I am child (pid:%d)\n", (int) getpid());
		close(fd1[0]);
		for (int i=0; i<1000000000; i++){
			if (i%10000000 == 0){
				//std::cout << i/1000000000. << std::endl;
			}
		}
		std::cout << "Done!" << std::endl;
		write(fd1[1], message, strlen(message) + 1);
		close(fd1[1]);
		exit(0);
	} else {
		printf("hello, I am parent of %d (pid:%d)\n", pID, (int) getpid());
		close(fd1[1]);
	}
	while (true) {
		int status;
		pid_t result = waitpid(pID, &status, WNOHANG);
		if (result == 0) {
		  std::cout << "Child Alive!" << std::endl;
		} else if (result == -1) {
		  std::cout << "Child Error!" << std::endl;
		} else {
		  std::cout << "Child Exited!" << std::endl;
		  break;
		}
	}
	char from_child[100];
	read(fd1[0], from_child, 100);
	printf("From Child: %s\n", from_child);
	close(fd1[0]);
}

std::vector<GLuint> get_textures_ids(){
	std::vector<GLuint> ids;
	char from_child[100];
	read(loading_texture_pipe[0], from_child, 100);
	printf("From Child: %s\n", from_child);
	close(loading_texture_pipe[0]);
	return ids;
}
