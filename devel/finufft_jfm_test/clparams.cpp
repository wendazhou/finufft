#include "clparams.h"

CLParams::CLParams(int argc, char* argv[])
{
    this->success = true; //let's be optimistic!

    //find the named and unnamed parameters checking for errors along the way
    for (int i = 1; i < argc; i++) {
        QuteString str = QuteString(argv[i]);
        if (str.startsWith("--")) {
            int ind2 = str.indexOf("=");
            QuteString name = str.mid(2,ind2-2);
            QuteString val = "";
            if (ind2 >= 0)
                val = str.mid(ind2+1);
            if (name.count()==0) {
                this->success = false;
                this->error_message = "Problem with parameter: " + str;
                return;
            }
            QuteString val2 = val;
            {
                this->named_parameters[name] = val2;
            }
        }
        else {
            //this->unnamed_parameters << str;
        }
    }
}
