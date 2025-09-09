#include <iostream>
#include <chrono>
#include <thread> //std::this_thread::sleep_for(std::chrono::milliseconds(50));
#include "matplotlibcpp.h"
#include <string.h>

#define MAZE_SIZE 100
#define NUMBER_OF_WALLS 300
#define STARTING_X 0
#define STARTING_Y 0
#define FINISH_X 99
#define FINISH_Y 99
#define NUMBER_OF_DIGIT 3
#define NUMBER_OF_DIGIT_FULL 3
#define theta 0.04

namespace plt = matplotlibcpp;

class my_queue{
    std::vector<std::vector<int>> data;
    int last_element = 0;
    int n;
    int real_size = 0;

    int get_next_index(int i){
        if (i+1 < real_size){
            return i+1;
        }else{
            return 0;
        }
    }
    public:

    my_queue(int my_n){
        n = my_n;
        data = std::vector<std::vector<int>>(n);
        for (int i=0; i<n; i++){
            data[i] = std::vector<int>(2);
            data[i][0] = 0;
            data[i][1] = 0;
        }
    }

    void add_element(int x, int y){
    	if (n > 0){
		    bool already_there = false;
		    for (int i=0; i<n; i++){
		        if (data[i][0] == x && data[i][1] == y){
		            already_there = true;
		        }
		    }
		    if (!already_there){
		        if (real_size < n){
		            real_size++;
		        }
		        last_element = get_next_index(last_element);
		        data[last_element][0] = x;
		        data[last_element][1] = y;
		    }
		}
    }

    int size(){
        return real_size;
    }

    std::vector<int> at(int i){
        int actual_i = get_next_index(last_element);
        for (int ii=0; ii<i; ii++){
            actual_i = get_next_index(actual_i);
        }
        return data[actual_i];
    }

    void print(){
        for (int i=0; i<n; i++){
            std::cout << "[" << data[i][0] << ", " << data[i][1] << "]";
        }
        std::cout << std::endl;
    }

};

class PQueue{
    #define multiplier 1000
    std::vector<std::vector<int>> data = std::vector<std::vector<int>>(0); // Inside Vector: [P, x, y, a]
    int max_size;
    public:
    PQueue(int my_max_size){
        max_size = my_max_size;
    }

    int P;
    std::vector<std::vector<int>> element = std::vector<std::vector<int>>(1);
    std::vector<std::vector<int>>::iterator ptr;
    void add(double double_P, int x, int y, int a){
        P = (int)(double_P*multiplier);
        element[0] = {P, x, y, a};
        ptr = data.begin();
        if (data.size() != 0){
            for (int i=0; i<data.size(); i++){
                if (data[i][0] < P){
                    advance(ptr, i);
                    copy(element.begin(), element.end(), inserter(data, ptr));
                    break;
                }
                if (i==data.size()-1){
                    data.push_back(element[0]);
                    break;
                }
            }
        }else{
            data.push_back(element[0]);
        }
        if (data.size()>max_size){
            data.erase(data.end());
        }
        // print_pqueue();
    }

    std::vector<int> result;
    std::vector<int> first(){
        result = data.front();
        data.erase(data.begin());
        // print_pqueue();
        return result;
    }


    void print_pqueue(){
        std::vector<std::vector<int>>::iterator ptr;
        std::cout << "PQueue: [";
        for (ptr = data.begin(); ptr < data.end(); ptr++){
            std::cout << "(" << (*ptr)[0] << ", " << (*ptr)[1] << ", " << (*ptr)[2] << ", " << (*ptr)[3] << "), ";
        }
        std::cout << "]" << std::endl;
    }

    bool not_empty(){
        if (data.size() > 0 ){
            return true;
        }
        return false;
    }

    int size(){
        return data.size();
    }
};

class Simple_Maze{

    double Q[MAZE_SIZE][MAZE_SIZE][4];
    std::vector<double> Model[MAZE_SIZE][MAZE_SIZE][4];
    bool Maze[MAZE_SIZE][MAZE_SIZE]; // true where there is a wall
    bool Observed_State[MAZE_SIZE][MAZE_SIZE];
    bool Observed_Action[MAZE_SIZE][MAZE_SIZE][4];
    bool Place_Already_Visited[MAZE_SIZE][MAZE_SIZE];

    int s[2];
    int s_prime[2];
    int state_before_planning[2];
    double reward;
    int action;

    int n;
    double epsilon;
    double alpha;
    double gamma;
    int num_of_episode = 0;
    double starting_epsilon;
    int zero_epsilon_episode;
    int random_actions;
    int iterations;
    int around[4][2] = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    int around_move[4] = {2, 3, 0, 1};
	
	std::vector<double> t, move;
	std::string name;
	
    void create_a_maze(){
        int rand_x, rand_y;
        for (int i=0; i<NUMBER_OF_WALLS; i++){
            do{
                rand_x = rand() % MAZE_SIZE;
                rand_y = rand() % MAZE_SIZE;
            }while(Maze[rand_x][rand_y]
                || (rand_x == STARTING_X && rand_y == STARTING_Y)
                || (rand_x == FINISH_X && rand_y == FINISH_Y));
            Maze[rand_x][rand_y] = true;
        }
    }
	
	std::string q_str(double v, int digit){
		std::string s = std::to_string(v);
		std::string zero = std::string("0");
		int missing_values = digit + 2 - s.size();
		if (missing_values > 0){
			for (int i=0; i<missing_values; i++){
				s = s + zero;
			}
		}else{
			for (int i=0; i<-missing_values; i++){
				s.pop_back();
			}
		}
		return s;
	}
	
    void print_maze(){
        std::cout << "/";
        for (int y=0; y<MAZE_SIZE*2-1; y++){
            std::cout << "-";
        }
        std::cout << "\\" << std::endl;
        for (int x=0; x<MAZE_SIZE; x++){
            std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (x == STARTING_X && y == STARTING_Y){
                    std::cout << "S|";
                }else if (x == FINISH_X && y == FINISH_Y){
                    std::cout << "F|";
                }else if (!Maze[x][y]){
                    std::cout << " |";
                }else{
                    std::cout << "#|";
                }
            }
            if (x!=MAZE_SIZE-1){
                std::cout << "\n|";
                for (int y=0; y<MAZE_SIZE*2-1; y++){
                    std::cout << "-";
                }
                std::cout << "|";
            }
            std::cout << "\n";
        }
        std::cout << "\\";
        for (int y=0; y<MAZE_SIZE*2-1; y++){
            std::cout << "-";
        }
        std::cout << "/" << std::endl;
    }

    void print_maze(int posx, int posy){
        std::cout << "/";
        for (int y=0; y<MAZE_SIZE*2-1; y++){
            std::cout << "-";
        }
        std::cout << "\\" << std::endl;
        for (int x=0; x<MAZE_SIZE; x++){
            std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (x ==  posx && y == posy){
                    std::cout << "@|";
                }else if (x == STARTING_X && y == STARTING_Y){
                    std::cout << "S|";
                }else if (x == FINISH_X && y == FINISH_Y){
                    std::cout << "F|";
                }else if (!Maze[x][y]){
                    std::cout << " |";
                }else{
                    std::cout << "#|";
                }
            }
            if (x!=MAZE_SIZE-1){
                std::cout << "\n|";
                for (int y=0; y<MAZE_SIZE*2-1; y++){
                    std::cout << "-";
                }
                std::cout << "|";
            }
            std::cout << "\n";
        }
        std::cout << "\\";
        for (int y=0; y<MAZE_SIZE*2-1; y++){
            std::cout << "-";
        }
        std::cout << "/" << std::endl;
    }
	
    void print_all(){
    	int best_action;
    	for (int i = 0; i<3; i++){
		    std::cout << "/";
		    if (i == 1){
            	for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT+2)+MAZE_SIZE-1; y++){
					std::cout << "-";
				}
            }else{
				for (int y=0; y<MAZE_SIZE*2-1; y++){
					std::cout << "-";
				}
			}
		    std::cout << "\\";
		}
		std::cout << "\n";
        for (int x=0; x<MAZE_SIZE; x++){
        	// ACTION
       		std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                best_action = find_best_action(x, y);
                if (!Maze[x][y]){
                    std::cout << best_action << "|";
                }else{
                    std::cout << "#|";
                }
            }
            // Q
            std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                best_action = find_best_action(x, y);
                if (!Maze[x][y]){
                    std::cout << q_str(Q[x][y][best_action], NUMBER_OF_DIGIT) << "|";
                }else{
                	for (int i=0; i < NUMBER_OF_DIGIT+2; i++){
                		std::cout << "#";
                	}
                	std::cout << "|";
                }
            }
            //OBSERVED
            std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (Observed_State[x][y]){
                    std::cout << "@|";
                }else{
                    std::cout << "#|";
                }
            }
            if (x!=MAZE_SIZE-1){
                std::cout << "\n";
                for (int i = 0; i<3; i++){
                	std::cout << "|";
                	if (i == 1){
                		for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT+2)+MAZE_SIZE-1; y++){
				            std::cout << "-";
				        }
                	}else{
				        for (int y=0; y<MAZE_SIZE*2-1; y++){
				            std::cout << "-";
				        }
				    }
		            std::cout << "|";
		        }
            }
            std::cout << "\n";
        }
        for (int i = 0; i<3; i++){
		    std::cout << "\\";
            if (i == 1){
            	for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT+2)+MAZE_SIZE-1; y++){
					std::cout << "-";
				}
            }else{
				for (int y=0; y<MAZE_SIZE*2-1; y++){
					std::cout << "-";
				}
			}
		    std::cout << "/";;
		}
		std::cout << "\n";
    }

    void print_Q(){
        std::cout << "/";
        for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT_FULL+2)*3+MAZE_SIZE-1; y++){
			std::cout << "-";
		}
        std::cout << "\\" << std::endl;
        for (int x=0; x<MAZE_SIZE; x++){
            std::cout << "|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (!Maze[x][y]){
                	for (int i=0; i < NUMBER_OF_DIGIT_FULL+2; i++){
                		std::cout << " ";
                	}
                    std::cout << q_str(Q[x][y][0], NUMBER_OF_DIGIT_FULL);
                    for (int i=0; i < NUMBER_OF_DIGIT_FULL+2; i++){
                		std::cout << " ";
                	}
                    std::cout << "|";
                }else{
                	for (int i=0; i < (NUMBER_OF_DIGIT_FULL+2)*3; i++){
                		std::cout << "#";
                	}
                	std::cout << "|";
                }
            }
            std::cout << "\n|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (!Maze[x][y]){
                    std::cout << q_str(Q[x][y][3], NUMBER_OF_DIGIT_FULL);
                    for (int i=0; i < NUMBER_OF_DIGIT_FULL+2; i++){
                		std::cout << " ";
                	}
                	std::cout << q_str(Q[x][y][1], NUMBER_OF_DIGIT_FULL);
                    std::cout << "|";
                }else{
                	for (int i=0; i < (NUMBER_OF_DIGIT_FULL+2)*3; i++){
                		std::cout << "#";
                	}
                	std::cout << "|";
                }
            }
            std::cout << "\n|";
            for (int y=0; y<MAZE_SIZE; y++){
                if (!Maze[x][y]){
                	for (int i=0; i < NUMBER_OF_DIGIT_FULL+2; i++){
                		std::cout << " ";
                	}
                    std::cout << q_str(Q[x][y][2], NUMBER_OF_DIGIT_FULL);
                    for (int i=0; i < NUMBER_OF_DIGIT_FULL+2; i++){
                		std::cout << " ";
                	}
                    std::cout << "|";
                }else{
                	for (int i=0; i < (NUMBER_OF_DIGIT_FULL+2)*3; i++){
                		std::cout << "#";
                	}
                	std::cout << "|";
                }
            }
            if (x!=MAZE_SIZE-1){
                std::cout << "\n";
                std::cout << "|";
                for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT_FULL+2)*3+MAZE_SIZE-1; y++){
					std::cout << "-";
				}
		        std::cout << "|";
            }
            std::cout << "\n";
        }
		std::cout << "\\";
        for (int y=0; y<MAZE_SIZE*(NUMBER_OF_DIGIT_FULL+2)*3+MAZE_SIZE-1; y++){
			std::cout << "-";
		}
		std::cout << "/";;
		std::cout << "\n";
    }

    double enviroment(int act){
        switch (act){
        case 0: // UP
            s_prime[0] = s[0]-1;
            s_prime[1] = s[1];
            break;
        case 1: // RIGHT
            s_prime[0] = s[0];
            s_prime[1] = s[1]+1;
            break;
        case 2: // DOWN
            s_prime[0] = s[0]+1;
            s_prime[1] = s[1];
            break;
        case 3: // LEFT
            s_prime[0] = s[0];
            s_prime[1] = s[1]-1;
            break;
        default:
            exit(0);
        }
        if (s_prime[0]<0 || s_prime[0]>=MAZE_SIZE || s_prime[1]<0 || s_prime[1]>=MAZE_SIZE || Maze[s_prime[0]][s_prime[1]]){
            s_prime[0] = s[0];
            s_prime[1] = s[1];
        }
        if (s_prime[0] == FINISH_X && s_prime[1] == FINISH_Y){
            return 1.;
        }else if (Place_Already_Visited[s_prime[0]][s_prime[1]]){
       		return -0.1;
       	}else{
            return 0.;
        }
    }

    int epsilon_greedy_action(int x, int y){
        double value = (rand() % 100001)/100000.;
        //std::cout << value << std::endl;
        if (value <= epsilon){
        	random_actions++;
            return rand() % 4;
        }else{
            return find_best_action(x, y);
        }
    }

    int find_best_action(int x, int y){
        double max_a_value = Q[x][y][0];
        int max_a = 0;
        for (int a=1; a<4; a++){
            if (Q[x][y][a] > max_a_value){
                max_a_value = Q[x][y][a];
                max_a = a;
            }
        }
        return max_a;
    }

    int select_already_observed__action(int x, int y){
        int a;
        do{
            a  = rand() % 4;
        }while(!Observed_Action[x][y][a]);
        return a;
    }

    public:
    Simple_Maze(std::string my_name, int my_n, int my_iterations, double my_epsilon=0.5, double my_alpha=0.1, double my_gamma=0.95){
        n = my_n;
        starting_epsilon = my_epsilon;
        epsilon = my_epsilon;
        alpha = my_alpha;
        gamma = my_gamma;
        zero_epsilon_episode = my_iterations;
        iterations = my_iterations;
        name = my_name;
        for (int x=0; x<MAZE_SIZE; x++){
            for (int y=0; y<MAZE_SIZE; y++){
                for (int a=0; a<4; a++){
                    Q[x][y][a] = 0;
                    Model[x][y][a] = std::vector<double>(3);
                    Observed_Action[x][y][a] = false;
                }
                Maze[x][y] = false;
                Observed_State[x][y] = false;
            }
        }
        create_a_maze();
        print_maze();
        t = std::vector<double>(iterations);
        move = std::vector<double>(iterations);
    }

    int run_episode(bool verbose){
    	random_actions = 0;
        s[0] = STARTING_X;
        s[1] = STARTING_Y;
        for (int x=0; x < MAZE_SIZE; x++){
        	for (int y=0; y < MAZE_SIZE; y++){
        		Place_Already_Visited[x][y] = false;
        	}
        }
        bool finished_episode = false;
        int iterations = 0;
        my_queue queue(n);
        while(!finished_episode){
        	Place_Already_Visited[s[0]][s[1]] = true;
            queue.add_element(s[0], s[1]);
            action = epsilon_greedy_action(s[0], s[1]);
            reward = enviroment(action);
            Q[s[0]][s[1]][action] += alpha*(reward+gamma*Q[s_prime[0]][s_prime[1]][find_best_action(s_prime[0], s_prime[1])]-Q[s[0]][s[1]][action]);
			
            Model[s[0]][s[1]][action][0] = s_prime[0];
            Model[s[0]][s[1]][action][1] = s_prime[1];
            Model[s[0]][s[1]][action][2] = reward;
            Observed_State[s[0]][s[1]] = true;
            Observed_Action[s[0]][s[1]][action] = true;
            if (reward > 0){
                finished_episode = true;
            }
            state_before_planning[0] = s_prime[0];
            state_before_planning[1] = s_prime[1];
            for (int i=0; i<queue.size(); i++){
                s[0] = queue.at(i)[0];
                s[1] = queue.at(i)[1];
                action = select_already_observed__action(s[0], s[1]);
                s_prime[0] = (int)(Model[s[0]][s[1]][action][0]);
                s_prime[1] = (int)(Model[s[0]][s[1]][action][1]);
                reward = Model[s[0]][s[1]][action][2];
                Q[s[0]][s[1]][action] += alpha*(reward+gamma*Q[s_prime[0]][s_prime[1]][find_best_action(s_prime[0], s_prime[1])]-Q[s[0]][s[1]][action]);
            }
            s[0] = state_before_planning[0];
            s[1] = state_before_planning[1];
            if (verbose){
                print_maze(s[0], s[1]);
                std:: cout << "Reward = " << reward << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            iterations ++;
        }
        epsilon = -starting_epsilon/(double)(zero_epsilon_episode)*num_of_episode + starting_epsilon;
        if (epsilon < 0){
        	epsilon = 0;
        }
        if (verbose){
        	print_all();
        	std:: cout << "[" << num_of_episode << "]Iteration in the episode: " << iterations << std::endl;
        	std::cout << "Random actions: " << random_actions << std::endl;
        }else{
            std:: cout << "[" << num_of_episode << "]Iteration in the episode: " << iterations << std::endl;;
        }
        //print_Q();
        num_of_episode++;
        return iterations;
    }
    
    int run_episode_prioritize_sweeping(bool verbose){
    	random_actions = 0;
        s[0] = STARTING_X;
        s[1] = STARTING_Y;
        for (int x=0; x < MAZE_SIZE; x++){
        	for (int y=0; y < MAZE_SIZE; y++){
        		Place_Already_Visited[x][y] = false;
        	}
        }
        bool finished_episode = false;
        int iterations = 0;
        PQueue pqueue = PQueue(n*2);
        double P;
        std::vector<int> element;
        int x, y;

        while(!finished_episode){
        	Place_Already_Visited[s[0]][s[1]] = true;
            action = epsilon_greedy_action(s[0], s[1]);
            reward = enviroment(action);
            P = abs(reward+gamma*Q[s_prime[0]][s_prime[1]][find_best_action(s_prime[0], s_prime[1])]-Q[s[0]][s[1]][action]);
            // std::cout << "Fuori:" << P << std::endl;
            if (P>theta){
                pqueue.add(P, s[0], s[1], action);
            }
            Model[s[0]][s[1]][action][0] = s_prime[0];
            Model[s[0]][s[1]][action][1] = s_prime[1];
            Model[s[0]][s[1]][action][2] = reward;
            Observed_State[s[0]][s[1]] = true;
            Observed_Action[s[0]][s[1]][action] = true;
            if (s_prime[0] == FINISH_X && s_prime[1] == FINISH_Y){
                finished_episode = true;
            }
            state_before_planning[0] = s_prime[0];
            state_before_planning[1] = s_prime[1];
            for (int i=0; i<n; i++){
                if (!pqueue.not_empty()){
                    break;
                }
                element = pqueue.first();
                s[0] = element[1];
                s[1] = element[2];
                action = element[3];;
                s_prime[0] = (int)(Model[s[0]][s[1]][action][0]);
                s_prime[1] = (int)(Model[s[0]][s[1]][action][1]);
                reward = Model[s[0]][s[1]][action][2];
                Q[s[0]][s[1]][action] += alpha*(reward+gamma*Q[s_prime[0]][s_prime[1]][find_best_action(s_prime[0], s_prime[1])]-Q[s[0]][s[1]][action]);
                for (int ii=0; ii<4; ii++){
                    x = s[0] + around[ii][0];
                    y = s[1] + around[ii][1];
                    action = around_move[ii];
                    if (x>=0 && y>=0 && x<MAZE_SIZE && y<MAZE_SIZE && Observed_Action[x][y][action]){
                        reward = Model[x][y][action][2];
                        P = abs(reward+gamma*Q[s[0]][s[1]][find_best_action(s[0], s[1])]-Q[x][y][action]);
                        // std::cout << "Dentro: " << P << std::endl;
                        if (P>theta){
                            pqueue.add(P, x, y, action);
                        }
                    }
                }
            }
            s[0] = state_before_planning[0];
            s[1] = state_before_planning[1];
            if (verbose){
                print_maze(s[0], s[1]);
                std::cout << "Reward = " << reward << std::endl;
                std::cout << "Size of Queue = " <<  pqueue.size() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            iterations ++;
            /*if (iterations % 1000 == 0){
                std::cout << "Size of Queue = " <<  pqueue.size() << std::endl;
            }*/
        }
        epsilon = -starting_epsilon/(double)(zero_epsilon_episode)*num_of_episode + starting_epsilon;
        if (epsilon < 0){
        	epsilon = 0;
        }
        if (verbose){
        	print_all();
        	std:: cout << "[" << num_of_episode << "]Iteration in the episode: " << iterations << std::endl;
        	std::cout << "Random actions: " << random_actions << "(epsilon = " << epsilon << ")" << std::endl;
        }else{
            std:: cout << "[" << num_of_episode << "]Iteration in the episode: " << iterations << std::endl;
        }
        //print_Q();
        num_of_episode++;
        return iterations;
    }

    void learn(bool verbose){
    	for (int iter=0; iter<iterations; iter++){
			t.at(iter) = iter;
		    move.at(iter) = run_episode(verbose);
    	}
    	plt::figure();
    	plt::title(name);
    	plt::xlabel("Episodes");
    	plt::ylabel("Step for Escape");
    	plt::plot(t, move, {{"label", "Moves"}});
    	plt::grid(true);
    	plt::save("./"+name+".png");
    	//print_Q();
    	std::cout << "Done! [" << move.at(iterations-1) << "]" << std::endl;
    }

    void learn_prioritize_sweeping(bool verbose){
    	for (int iter=0; iter<iterations; iter++){
			t.at(iter) = iter;
		    move.at(iter) = run_episode_prioritize_sweeping(verbose);
    	}
    	plt::figure();
    	plt::title(name);
    	plt::xlabel("Episodes");
    	plt::ylabel("Step for Escape");
    	plt::plot(t, move, {{"label", "Moves"}});
    	plt::grid(true);
    	plt::save("./"+name+".png");
    	//print_Q();
    	std::cout << "Done! [" << move.at(iterations-1) << "]" << std::endl;
    }
	
	void show_what_learn(){
		std::cout << "Starting showing the path founded in\n3..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "2..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "1..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		run_episode(true);
	}
	
	void show_what_learn_prioritize_sweeping(){
		std::cout << "Starting showing the path founded in\n3..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "2..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "1..." << std::endl;
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		run_episode_prioritize_sweeping(true);
	}

    void human_control(){
        s[0] = STARTING_X;
        s[1] = STARTING_Y;
        while(true){
            std::cout << "Action [0=UP, 1=RIGHT, 2=DOWN, 3=LEFT] = ";
            std::cin >> action;
            reward = enviroment(action);
            print_maze(s_prime[0], s_prime[1]);
            s[0] = s_prime[0];
            s[1] = s_prime[1];
            std:: cout << "Reward = " << reward << std::endl;
        }
    }
};

#define iteration 3000
int main(){
    srand(1626);
    Simple_Maze SM = Simple_Maze("n=20 my_queue", 20, iteration);
    SM.learn(false);
    srand(1626);
    Simple_Maze SM_prioritize_sweeping = Simple_Maze("n=20 prioritize_sweeping", 20, iteration);
    SM_prioritize_sweeping.learn_prioritize_sweeping(false);
    //SM.show_what_learn();
    //SM.show_what_learn_prioritize_sweeping();
}
