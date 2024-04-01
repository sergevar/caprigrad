#include <thread>
#include <atomic>
#include <iostream>
#include <termios.h>
#include <unistd.h>

#include "../src/ffml/ffml.h"

struct KeyboardEvent {
    char key;
    bool pressed;
    int id;
};


// void changemode(int);
// int  kbhit(void);

// run your keyboard listening in this thread

class KeyboardListener {
private:
    int lastKbEventId = 0;

    std::atomic<KeyboardEvent> kb_event;    // flag to control when to exit the application
    std::thread * t2;

    void keyboard_thread() 
    {
        char c;
        while(true)
        {
            if(kbhit()) 
            {
                c = getchar();

                int random_id = rand() % 1000000;

                kb_event.store({
                    c, true, random_id
                });

                // printf("key: %c\n", c);

                switch(c) {
                    // case ']':
                    //     LR_atomic.store(LR_atomic.load() + LR_INITIAL);
                    //     printf("LR: %f\n", LR_atomic.load());
                    //     break;
                    // case '[':
                    //     LR_atomic.store(LR_atomic.load() - LR_INITIAL);
                    //     printf("LR: %f\n", LR_atomic.load());
                    //     break;
                    default:
                        break;
                }

                // chill
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void changemode(int dir)
    {
        static struct termios oldt, newt;
                        
        if ( dir == 1 )
        {
            tcgetattr( STDIN_FILENO, &oldt);
            newt = oldt;
            newt.c_lflag &= ~( ICANON | ECHO );
            tcsetattr( STDIN_FILENO, TCSANOW, &newt);
        }
        else
            tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
    }

    int kbhit (void)
    {
        struct timeval tv;
        fd_set rdfs;

        tv.tv_sec = 0;
        tv.tv_usec = 0;

        FD_ZERO(&rdfs);
        FD_SET (STDIN_FILENO, &rdfs);

        select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
        return FD_ISSET(STDIN_FILENO, &rdfs);
    }

public:

    void start() {
        changemode(1);
        
        // create the thread for keyboard listening
        t2 = new std::thread(&KeyboardListener::keyboard_thread, this);
    }

    char check() {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto ev = kb_event.load();
        if (ev.id != lastKbEventId) {
            lastKbEventId = ev.id;

            if (ev.pressed) {
                return ev.key;
            } else {
                return 0;
            }
        } else {
            return 0;
        }
    }

    ~KeyboardListener() {
        changemode(0);
        t2->join();
        delete t2;
    }
 

};

class KeyboardListenerGrad {
private:
    KeyboardListener kl;
    ffml_cgraph * cgraph;
    std::unordered_map<char, std::function<void()>> callbacks;
public:
    KeyboardListenerGrad(ffml_cgraph * _cgraph) {
        cgraph = _cgraph;
        kl.start();
    }

    void check() {
        char c = kl.check();

        if (c == 0) return;

        if (callbacks.find(c) != callbacks.end()) {
            callbacks[c]();
        } else {
            switch(c) {
                case 'q':
                    printf("Quitting...\n");
                    exit(0);
                    break;
                case 'D':
                    printf("Dumping the graph...");
                    ffml_debug_print_cgraph_shapes(this->cgraph);
                    ffml_debug_print_cgraph_data(this->cgraph);
                    exit(0);
                    break;
                case 'p':
                    this->pause();
                    break;
                default:
                    break;
            }
        }
    }

    void pause() {
        printf("Paused. Press 'p' to continue.\n");
        while(true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            char c = kl.check();
            if (c == 'p') {
                break;
            }
        }
    }

    // attach a function to be called
    void attach(char key, std::function<void()> callback) {
        callbacks[key] = callback;
    }
};
