#include "server.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <nlohmann/json.hpp>
#include <queue>
#include <mutex>

#include "routes.h"

struct IncomingMessage {
    std::string request_id;
    std::string command;
    nlohmann::json data;
};

struct OutgoingMessage {
    std::string request_id;
    nlohmann::json data;
};

std::queue<IncomingMessage> incoming_messages;
std::queue<OutgoingMessage> outgoing_messages;

std::mutex mtx_incoming;
std::mutex mtx_outgoing;

SharedMemory sharedMemory;

void addToIncomingMessages(std::string request_id, std::string command, nlohmann::json data) {
    std::lock_guard<std::mutex> lock(mtx_incoming);  // lock the mutex

    incoming_messages.push({
        request_id, command, data
    });
}

void addToOutgoingMessages(std::string request_id, nlohmann::json data) {
    std::lock_guard<std::mutex> lock(mtx_outgoing);  // lock the mutex

    outgoing_messages.push({
        request_id, data
    });
}

using tcp = boost::asio::ip::tcp;
namespace websocket = boost::beast::websocket;

void fail(boost::system::error_code ec, char const* what)
{
    std::cerr << what << ": " << ec.message() << "\n";
}

class Session : public std::enable_shared_from_this<Session> {
    websocket::stream<tcp::socket> ws_;
    boost::beast::multi_buffer buffer_;

public:
    std::queue<std::string> write_msgs_;

    explicit Session(tcp::socket socket) : ws_(std::move(socket)) {}

    void run() {
        ws_.async_accept(
            boost::asio::bind_executor(
                ws_.get_executor(),
                std::bind(
                    &Session::on_accept,
                    shared_from_this(),
                    std::placeholders::_1)));
    }

    void send_string(const std::string& request_id, const std::string& data) {
        // printf("send_string: %s\n", data.c_str());

        write_string(request_id, data);
    }

private:

    void on_accept(boost::system::error_code ec) {
        if(ec)
            return fail(ec, "accept");

        do_read();
    }

    void do_read() {
        if(!ws_.next_layer().is_open()) {
            return;
        }
        ws_.async_read(
            buffer_,
            std::bind(
                &Session::on_read,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2));
    }

    void on_read(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // printf("on_read\n");

        if(ec) {
            if(ec == boost::asio::error::operation_aborted || 
            ec == boost::beast::websocket::error::closed) {
                std::cerr << "Connection closed" << std::endl;
                return;
            } else {
                fail(ec, "read");
            }
        }

        // Parsing the incoming JSON
        nlohmann::json receivedJson;
        std::string message = boost::beast::buffers_to_string(buffer_.data());
        try {
            receivedJson = nlohmann::json::parse(message);
        } catch (nlohmann::json::exception&) {
            // If the parsing throws an exception (i.e., invalid JSON), simply consume it and ignore
            printf("Invalid JSON received\n");
            printf("buffer: %s\n", boost::beast::buffers_to_string(buffer_.data()).c_str());
            buffer_.consume(buffer_.size());
            do_read();
            return;
        }

        // printf("receivedJson: %s\n", receivedJson.dump().c_str());

        std::string request_id = receivedJson["request_id"];
        std::string command = receivedJson["command"];
        nlohmann::json data = receivedJson["data"];

        // Route the command
        // std::string response = route_command(command, data, cgraph);

        // Send the response
        // write_string(response);

        addToIncomingMessages(request_id, command, data);

        buffer_.consume(buffer_.size());

        do_read();
    }

    std::queue<std::string> pieces;

    void prepare_pieces(const std::string request_id, const std::string & s) {
        const int MAX = 2048;
        // split the string into chunks and save into the queue
        for (size_t i = 0; i < s.size(); i += MAX) {
            std::string pushdata = request_id + (i == 0 ? "|S" : "|M") + s.substr(i, MAX);
            pieces.push(pushdata);
        }
        pieces.push(request_id + "|E");
    }

    void write_string(const std::string request_id, const std::string & s) {
        prepare_pieces(request_id, s);
        write_piece();
    }

    void write_piece() {
        if (pieces.empty()) {
            return;
        }
        std::string piece = pieces.front();
        pieces.pop();
        auto self(shared_from_this());
        ws_.async_write(boost::asio::buffer(piece),
            [this, self](boost::system::error_code ec, std::size_t bytes) {
                if (!ec) {
                    write_piece();
                } else {
                    std::cerr << "write failed: " << ec.message() << std::endl;
                }
            });
    }

    // void write_string(std::string s) {
    //     ws_.async_write(
    //         boost::asio::buffer(s),
    //         std::bind(
    //             &Session::on_write,
    //             shared_from_this(),
    //             std::placeholders::_1,
    //             std::placeholders::_2));
    // }
    
    void on_write(boost::system::error_code ec,
                  std::size_t bytes_transferred) {
        if(ec)
            return fail(ec, "write");

        // printf("on_write\n");
    }
};

// Global variable of Session.
// std::shared_ptr<Session> globalSession;
std::vector<std::shared_ptr<Session>> sessions;

std::shared_ptr<Session> lastSession() {
    if (sessions.size() == 0) {
        return nullptr;
    }
    return sessions[sessions.size() - 1];
}

class Listener : public std::enable_shared_from_this<Listener> {
    tcp::acceptor acceptor_;
    tcp::socket socket_;

public:
    Listener(
        boost::asio::io_context& ioc,
        tcp::endpoint endpoint)
        : acceptor_(ioc)
        , socket_(ioc)
    {
        boost::system::error_code ec;

        acceptor_.open(endpoint.protocol(), ec);
        acceptor_.set_option(boost::asio::socket_base::reuse_address(true), ec);
        acceptor_.bind(endpoint, ec);
        acceptor_.listen(
            boost::asio::socket_base::max_listen_connections, ec);

        if(ec)
            fail(ec, "listen");
    }

    // Start accepting incoming connections
    void run()
    {
        if(! acceptor_.is_open())
            return;
        do_accept();
    }

    void do_accept()
    {
        acceptor_.async_accept(
            socket_,
            std::bind(
                &Listener::on_accept,
                shared_from_this(),
                std::placeholders::_1));
    }

    void on_accept(boost::system::error_code ec)
    {
        if(ec)
        {
            fail(ec, "accept");
        }
        else
        {
            // create the Session, store it globally & run it
            // globalSession = std::make_shared<Session>(std::move(socket_));
            // globalSession->run();

            std::shared_ptr<Session> newSession = std::make_shared<Session>(std::move(socket_));
            sessions.push_back(newSession);
            newSession->run();


            printf("on_accept - new session assigned--------------\n");

            // Accept the next connection
            do_accept();
        }
    }
};

//------------------------------------------------------------------------------

void server_thread()
{
    auto const address = boost::asio::ip::make_address("127.0.0.1");
    auto const port = static_cast<unsigned short>(std::atoi("8889"));
    
    boost::asio::io_context ioc{1};

    std::make_shared<Listener>(ioc, tcp::endpoint{address, port})->run();

    ioc.run();
}

void check_incoming_messages() {
    std::lock_guard<std::mutex> lock(mtx_incoming);  // lock the mutex

    while (!incoming_messages.empty()) {
        // printf("thread_incoming_messages\n");

        IncomingMessage msg = incoming_messages.front();
        incoming_messages.pop();

        std::string command = msg.command;
        nlohmann::json data = msg.data;

        // Route the command
        nlohmann::json response = route_command(command, data, sharedMemory);

        // Send the response
        addToOutgoingMessages(msg.request_id, response);
    }
}

void check_outgoing_messages() {
    std::lock_guard<std::mutex> lock(mtx_outgoing);  // lock the mutex

    while (!outgoing_messages.empty()) {
        OutgoingMessage msg = outgoing_messages.front();
        outgoing_messages.pop();

        nlohmann::json data = msg.data;

        // printf("thread_outgoing_messages!\n");
        // printf("data: %s\n", data.dump().c_str());

        if (lastSession() != nullptr) {
            // nlohmann::json j;
            // j["request_id"] = msg.request_id;
            // j["data"] = data;

            lastSession()->send_string(msg.request_id, data.dump());
        }
    }
}

void thread_incoming_messages() {
    while (true) {
        check_incoming_messages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void thread_outgoing_messages() {
    while (true) {
        check_outgoing_messages();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

std::thread * t_server;
std::thread * t_incoming_messages;
std::thread * t_outgoing_messages;
void server_start(ffml_cgraph * _cgraph) {
    sharedMemory.cgraph = _cgraph;

    t_server = new std::thread(server_thread);
    t_incoming_messages = new std::thread(thread_incoming_messages);
    t_outgoing_messages = new std::thread(thread_outgoing_messages);
}

std::string randomRequestId() {
    std::string s = "";
    for (int i = 0; i < 10; i++) {
        s += std::to_string(rand() % 10);
    }
    return s;
}

void server_push_string(const std::string& data) {
    addToOutgoingMessages(randomRequestId(), data);
}

void server_push_object(const nlohmann::json& data) {
    addToOutgoingMessages(randomRequestId(), data);
}

void server_push_event(const std::string eventType, const nlohmann::json& data) {
    nlohmann::json j;
    j["event"] = eventType;
    j["data"] = data;

    addToOutgoingMessages(randomRequestId(), j);
}

void server_loop_interrupt() {
    while (sharedMemory.should_pause) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}