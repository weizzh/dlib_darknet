#include "darknet.h"

#include <sys/time.h>
#include <assert.h>
#include "string.h"


char *result_show=" ";

char *datacfg ="../cfg/eye.data";
char *cfgfile ="../cfg/eye.cfg";
char *weightfile ="../data/eye_1000.weights";
char **names;
network *net;
int top;
void initial_network(){
	net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");  
    names = get_labels(name_list);
    top = option_find_int(options, "top", 1);
}
int predict_class(float *X){
	float *predictions = network_predict(net, X);
    int index = predictions[0]>predictions[1]? 0 :1;
//	printf("predictions: %f %f\n", predictions[0], predictions[1]);
//	top_k(predictions, net->outputs, top, indexes);
	return index;

}