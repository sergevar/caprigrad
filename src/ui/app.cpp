#include <QApplication>
#include <QTextEdit>
#include <QSplitter>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QPushButton>
#include <QTimer>
#include <QMainWindow>
#include <QBrush>
#include <QGraphicsSceneHoverEvent>
#include <QTreeView>
#include <QStandardItemModel>
#include <QAbstractItemModel>
#include <QStyledItemDelegate>

#include "app.h"

#include <iostream>

using namespace std;

#include <thread>


#include <QPushButton>
#include <QStyledItemDelegate>
#include <QApplication>

std::string pad(std::string s, int n) {
    while(s.size() < n) {
        s += " ";
    }
    return s;
}

class ButtonDelegate : public QStyledItemDelegate
{
public:
    explicit ButtonDelegate(QObject *parent = 0)
        : QStyledItemDelegate(parent)
    {}

    void paint(QPainter *painter, const QStyleOptionViewItem &option, const QModelIndex &index) const override
    {
        if (index.column() == 1) { // Adjust this to the column where you want to add your button
            QPushButton button("Button Name"); // You can replace this with your button
            button.setStyleSheet("QPushButton { background-color: rgb(255,0,0); }");
            painter->save();

            painter->setRenderHint(QPainter::Antialiasing, true);
            painter->setRenderHint(QPainter::SmoothPixmapTransform, true);

            painter->drawPixmap(option.rect.x(), option.rect.y(), button.icon().pixmap(24, 24));

            painter->restore();
        } else {
            QStyledItemDelegate::paint(painter, option, index);
        }
    }

    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const override
    {
        if (index.column() == 1) // Adjust this to the column where you want to add your button
            return new QPushButton("Button Name", parent); // You can replace this with your button

        return QStyledItemDelegate::createEditor(parent, option, index);
    }
};

struct Item
{
    int row;
    Item* parent;
    std::string txt;
    // List to hold child Items
    QList<Item*> children;
};

class MyModel : public QAbstractItemModel
{
public:

    ffml_cgraph* cgraph;

    Item* root;

    MyModel(ffml_cgraph * _cgraph, QObject *parent = nullptr)
        : QAbstractItemModel(parent), cgraph(_cgraph) {

        root = new Item;
        
        // populate the root item with children, e.g.:
        for (int i = 0; i < _cgraph->n_nodes; i++) {
            Item *childItem = new Item;
            childItem->row = i;
            childItem->parent = root;


            // create children items for tensor details
            Item* tensorNameItem = new Item;
            Item* tensorShapeItem = new Item;
            tensorNameItem->parent = childItem;
            tensorShapeItem->parent = childItem;
            tensorNameItem->row = i;
            tensorShapeItem->row = i;

            childItem->children.append(tensorNameItem);
            childItem->children.append(tensorShapeItem);



            root->children.append(childItem);
        }
    }

    QModelIndex index(int row, int column, const QModelIndex &parent = QModelIndex()) const override
    {
        if (!hasIndex(row, column, parent))
            return QModelIndex();

        Item* parentItem;

        if(!parent.isValid()) {
            parentItem = root;
        } else {
            parentItem = static_cast<Item*>(parent.internalPointer());
        }

        if (!parentItem->children.isEmpty() && row < parentItem->children.count()) {
            return createIndex(row, column, parentItem->children[row]);
        }

        return QModelIndex();


        // if (row < parentItem->children.count()) {
        //     Item* childItem = parentItem->children[row];
        //     return createIndex(row, column, childItem);
        // }

        return QModelIndex();
    }

    QModelIndex parent(const QModelIndex &index) const override
    {
        if(!index.isValid())
            return QModelIndex();

        Item* childItem = static_cast<Item*>(index.internalPointer());
        Item* parentItem = childItem->parent;

        if (parentItem == root)
            return QModelIndex();

        return createIndex(parentItem->row, 0, parentItem);

        // // For children items, return index of the string
        // if(index.internalPointer() != nullptr) {
        //     return createIndex(index.row(), 0, nullptr);
        // }

        // return QModelIndex(); // Top-level items have no parent
    }

    // int rowCount(const QModelIndex &parent = QModelIndex()) const override
    // {
    //     if(!parent.isValid())
    //         // return stringList.count(); // number of strings
    //         return cgraph->n_items;

    //     return 2;

    //     const int TENSOR_ROW_COUNT = 2;
    //     if(parent.internalPointer() != nullptr) {
    //         Item* parentItem = static_cast<Item*>(parent.internalPointer());
    //         // return parentItem->children.count();
    //         return TENSOR_ROW_COUNT;
    //     }

    //     return 0; // characters have no children
    // }

    int rowCount(const QModelIndex &parent = QModelIndex()) const override
    {
        Item* parentItem = !parent.isValid() ? root : static_cast<Item*>(parent.internalPointer());
        return parentItem->children.count();
    }

    int columnCount(const QModelIndex &parent = QModelIndex()) const override
    {
        return 1; // We're using a simple tree with single-column items
    }
    
    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override
    {
        // printf("asked for data\n");
        // printf("index: %d %d\n", index.row(), index.column());
        // printf("role: %d\n", role);
        // printf("parent: %d %d\n", index.parent().row(), index.parent().column());
        // printf("parent valid: %d\n", index.parent().isValid());

        if (!index.isValid() || role != Qt::DisplayRole)
            return QVariant();
        
        Item* item = static_cast<Item*>(index.internalPointer());

        if(! index.parent().isValid()) {
            // this is a parent
            ffml_tensor* tensor = cgraph->nodes[index.row()];
            std::string accum = "";

            // add memory address hex
            accum += pad(std::to_string((uint64_t)tensor), 30);

            accum += pad(std::string(tensor->name), 30);

            std::string shape = "";
            for(int i = 0; i < tensor->n_dims; i++) {
                shape += std::to_string(tensor->ne[i]);
                if(i != tensor->n_dims - 1)
                    shape += ", ";
            }
            shape = "(" + shape + ") = " + std::to_string(tensor->nelem);

            accum += pad(shape, 20);

            // src0
            if(tensor->src0 != nullptr) {
                accum += pad("src0: " + std::string(tensor->src0->name), 30);
            } else {
                accum += pad("", 30);
            }

            // src1
            if(tensor->src1 != nullptr) {
                accum += pad("src1: " + std::string(tensor->src1->name), 30);
            } else {
                accum += pad("", 30);
            }

            return QString(accum.c_str());
        } else {
            // this is a child
            ffml_tensor* tensor = cgraph->nodes[item->row];

            const int TENSOR_NAME_ROW = 0;
            const int TENSOR_SHAPE_ROW = 1;

            if(index.row() == TENSOR_NAME_ROW) {
                std::string accum = "";

                // add memory address hex
                accum += pad(std::to_string((uint64_t)tensor), 20);

                // src0
                if(tensor->src0 != nullptr) {
                    accum += pad("src0: " + std::string(tensor->src0->name), 20);
                } else {
                    accum += pad("", 20);
                }

                // src1
                if(tensor->src1 != nullptr) {
                    accum += pad("src1: " + std::string(tensor->src1->name), 20);
                } else {
                    accum += pad("", 20);
                }

                accum += pad(std::string(tensor->name), 20);

                std::string shape = "";
                for(int i = 0; i < tensor->n_dims; i++) {
                    shape += std::to_string(tensor->ne[i]);
                    if(i != tensor->n_dims - 1)
                        shape += ", ";
                }
                shape = "(" + shape + ")";

                accum += pad(shape, 20);

                return QString(accum.c_str());
            } else if (index.row() == TENSOR_SHAPE_ROW) {
                std::string shape = "";
                for(int i = 0; i < tensor->n_dims; i++) {
                    shape += std::to_string(tensor->ne[i]);
                    if(i != tensor->n_dims - 1)
                        shape += ", ";
                }
                shape = "(" + shape + ")";
                return QString(("shape: " + shape).c_str());
            }
        }
        
        return QVariant();
    }

};

void app_thread(ffml_cgraph * cgraph) {
    int argc = 0;
    char** argv = nullptr;
    QApplication app = QApplication(argc, argv);

    QMainWindow window;
    window.resize(2200, 1400);
    window.setWindowTitle("CapricornGrad");





    auto splitter = new QSplitter();

    // Define a Model  
    MyModel model(cgraph);

    QFont font("Monospace");
    font.setStyleHint(QFont::TypeWriter);

    // Add some items to the Model
    QTreeView * treeView = new QTreeView();
    treeView->setModel(&model);
    treeView->setFont(font);
    // treeView->setItemDelegate(new ButtonDelegate());
    // treeView->show();


    // auto logState = new map<string, string>;

    // QTextEdit* logTextEdit = new QTextEdit();
    // logTextEdit->setPlainText("Hello, World!");
    // logTextEdit->setReadOnly(true);

    // QGraphicsScene scene;

    // auto brainGraphicsItem = new BrainGraphicsItem(brain, logTextEdit, logState);

    // std::cout << "Drawing view" << std::endl;

    // scene.addItem(brainGraphicsItem);

    // auto view = new QGraphicsView(&scene);

    // // background to black
    // view->setBackgroundBrush(QBrush(Qt::black));
    // view->setMouseTracking(true);

    // on app quit, exit
    QObject::connect(&app, &QApplication::aboutToQuit, [&](){
        std::cout << "Quitting" << std::endl;
        exit(0);
    });

    // on window close, exit
    QObject::connect(&app, &QApplication::lastWindowClosed, [&](){
        std::cout << "Quitting" << std::endl;
        exit(0);
    });

    // auto button1 = new QPushButton("Hello, World!");
    auto button2 = new QPushButton("Goodbye, World!");

    // std::cout << "Connecting button1" << std::endl;

    // // Connect button1's clicked signal to a lambda function
    // QObject::connect(button1, &QPushButton::clicked, [&](){
    //     button1->setText("Hello, World! (clicked)");
    // });
    // exit button
    QObject::connect(button2, &QPushButton::clicked, qApp, &QApplication::quit);

    // QTimer timer;
    // QObject::connect(&timer, &QTimer::timeout, [&](){
    //     // collect inputs
    //     // auto inputs = world->getPlayerInputs();

    //     // int xx = brain->timestep % BRAIN_WIDTH;
    //     // int yy = (brain->timestep / BRAIN_WIDTH) % BRAIN_HEIGHT;
    //     int xx = 5;
    //     int yy = 5;

    //     Grid<float> inputs = Grid<float>(BRAIN_WIDTH, BRAIN_HEIGHT);
    //     for (int x = 0; x < BRAIN_WIDTH; x++) { 
    //         for (int y = 0; y < BRAIN_HEIGHT; y++) {
    //             inputs(x, y) = 0.0f;

    //             if (brain->timestep % 5 != 0) continue;

    //             // if (x == xx && y == yy && brain->timestep % 2 == 0) {
    //             if (x == y /*|| x+1 == y*/) {
    //                 inputs(x, y) = 0.4f;
    //             }
    //         }
    //     }

    //     Grid<float> outputs = brain->step(inputs);

    //     // world->step(outputs);

    //     if (brain->timestep % REPAINT_EVERY_N_TIMESTEPS == 0) {
    //         // std::cout << "Repainting" << std::endl;

    //         brain->processAll();

    //         brainGraphicsItem->update();
    //         // logTextEdit->setPlainText(QString::fromStdString("Timestep: " + std::to_string(brain->timestep)));

    //         // show log
    //         std::string logString = "";
    //         for (auto it = logState->begin(); it != logState->end(); it++) {
    //             logString += it->first + ": " + it->second + "\n";
    //         }
    //         logTextEdit->setPlainText(QString::fromStdString(logString));
    //     }
    // });
    // timer.start(TIMESTEP_TIMER_MS);

    // QTimer secondsTimer;
    // // calculate how many timesteps per second it's able to run
    // int timestepsPerSecond = 0;
    // int lastTimestep = 0;
    // QObject::connect(&secondsTimer, &QTimer::timeout, [&](){
    //     int timestepsThisSecond = brain->timestep - lastTimestep;
    //     lastTimestep = brain->timestep;
    //     timestepsPerSecond = timestepsThisSecond;
    //     // logTextEdit->append(QString::fromStdString("Timesteps per second: " + std::to_string(timestepsPerSecond)));
    //     logState->operator[]("Timesteps per second") = std::to_string(timestepsPerSecond);
    // });
    // secondsTimer.start(1000);

    window.setCentralWidget(splitter);

    // auto splitterRight = new QSplitter(Qt::Vertical);
    // splitterRight->addWidget(button2);
    // splitterRight->addWidget(logTextEdit);

    // splitter->addWidget(view);
    splitter->addWidget(treeView);
    splitter->addWidget(button2);

    // std::cout << "Showing window" << std::endl;

    window.show();

    // std::cout << "Running app" << std::endl;

    // world ui

    // worldui_init(world);

    app.exec();
}

std::thread * t_ui;

void app(ffml_cgraph * cgraph) {
    t_ui = new std::thread(app_thread, cgraph);
}