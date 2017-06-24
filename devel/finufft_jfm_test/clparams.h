#ifndef CLPARAMS_H
#define CLPARAMS_H

#include "qute.h"

class CLParams {
public:
    CLParams(int argc, char* argv[]);
    QuteStringMap named_parameters;
    QuteStringList unnamed_parameters;
    bool success;
    QuteString error_message;
};

#endif // CLPARAMS_H

