#include <ncurses.h>

int main() {
    int current = 0;
    int mala = 0;

    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);

    while (true) {
        clear();

        mvprintw(1, 2, "=== | ===");
        mvprintw(3, 2, "Big     : %d", mala);
        mvprintw(4, 2, "Current : %d", current);

        // mvprintw(7, 2, "Controls:");
        // mvprintw(8, 2, "Any Key = +1 Count");
        // mvprintw(9, 2, "c = Reset Current Count");
        // mvprintw(10, 2, "m = Reset Mala");
        // mvprintw(11, 2, "b = Reset Both");
        // mvprintw(12, 2, "q = Quit");

        refresh();

        int ch = getch();

        if (ch == 'r')
            break;
        else if (ch == 'q')
            current = 0;
        else if (ch == 'w')
            mala = 0;
        else if (ch == 'e') {
            current = 0;
            mala = 0;
        }
        else {
            current++;

            if (current >= 108) {
                mala++;
                current = 0;
            }
        }
    }

    endwin();
    return 0;
}