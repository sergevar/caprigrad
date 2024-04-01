#include <iostream>

enum class Color
{
    FG_RED = 31,
    FG_GREEN = 32,
    FG_YELLOW = 33,
    FG_BLUE = 34,
    FG_MAGENTA = 35,
    FG_CYAN = 36,
    FG_WHITE = 37,
    FG_RESET = 0,
    BG_RED = 41,
    BG_GREEN = 42,
    BG_YELLOW = 43,
    BG_BLUE = 44,
    BG_MAGENTA = 45,
    BG_CYAN = 46,
    BG_WHITE = 47,
    BG_RESET = 49
};

void setColor(Color color)
{
    std::cout << "\033[" << static_cast<int>(color) << "m";
}

void resetColor()
{
    setColor(Color::FG_RESET);
    setColor(Color::BG_RESET);
}

void printWithColor(std::string text, Color color)
{
    setColor(color);
    std::cout << text;
    resetColor();
}

void printWithColor(std::string text, Color color, Color background)
{
    setColor(color);
    setColor(background);
    std::cout << text;
    resetColor();
}

void printWithBackground(std::string text, Color background)
{
    setColor(background);
    std::cout << text;
    resetColor();
}