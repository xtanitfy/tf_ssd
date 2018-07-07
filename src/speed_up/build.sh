gcc -I /usr/include/python2.7/ -fPIC -shared -lstdc++ speed_up.c match_boxes.c -o speed_up.so
mv speed_up.so ..
