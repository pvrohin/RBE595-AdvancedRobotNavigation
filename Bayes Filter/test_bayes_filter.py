#Initial state of the door is unknown
bel_x0_open = 0.5
bel_x0_closed = 0.5

#Intitialize the measurement model
p_zt_open_xt_open = 0.6
p_zt_open_xt_closed = 0.2
p_zt_closed_xt_open = 0.4
p_zt_closed_xt_closed = 0.8

#Initialize the action model
p_xt_open_ut_push_xt_1_open = 1
p_xt_closed_ut_push_xt_1_open = 0
p_xt_open_ut_push_xt_1_closed = 0.8
p_xt_closed_ut_push_xt_1_closed = 0.2

p_xt_open_ut_do_nothing_xt_1_open = 1
p_xt_closed_ut_do_nothing_xt_1_open = 0
p_xt_open_ut_do_nothing_xt_1_closed = 0
p_xt_closed_ut_do_nothing_xt_1_closed = 1

def prediction(action):
    global bel_x0_open,bel_x0_closed
    if(action=="do_nothing"):
        bel_bar_x1_open = (p_xt_open_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_open_ut_do_nothing_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_closed_ut_do_nothing_xt_1_closed*bel_x0_closed)
    if(action=="open"):
        bel_bar_x1_open = (p_xt_open_ut_push_xt_1_open*bel_x0_open) + (p_xt_open_ut_push_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_push_xt_1_open*bel_x0_open) + (p_xt_closed_ut_push_xt_1_closed*bel_x0_closed)

    return bel_bar_x1_open,bel_bar_x1_closed

def correction(bel_bar_x1_open,bel_bar_x1_closed,measurement):
    global bel_x0_open,bel_x0_closed
    if(measurement=="open"):
        bel_x1_open = p_zt_open_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_open_xt_closed*bel_bar_x1_closed

    if(measurement=="closed"):
        bel_x1_open = p_zt_closed_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_closed_xt_closed*bel_bar_x1_closed

    norm = 1/(bel_x1_open+bel_x1_closed)

    bel_x1_open = norm*bel_x1_open

    bel_x1_closed = norm*bel_x1_closed

    bel_x0_open = bel_x1_open
    bel_x0_closed = bel_x1_closed
        
    return bel_x0_open,bel_x0_closed

def bayes_filter(action,measurement):
    global bel_x0_open,bel_x0_closed
    bel_bar_x1_open,bel_bar_x1_closed = prediction(action)
    bel_x0_open,bel_x0_closed = correction(bel_bar_x1_open,bel_bar_x1_closed,measurement)
    print("Probability the door is open is ",bel_x0_open)
    print("Probability the door is closed is ",bel_x0_closed)

def main():
    # actions = ["do_nothing","open","do_nothing","open","do_nothing"]
    # measurement = ["closed","closed","closed","open","open"]

    actions = ["open","open","open","open","open"]
    measurement = ["closed","closed","closed","closed","closed"]

    for i in range(0,5):
         print("Action",actions[i])
         print("Measurement",measurement[i])
         bayes_filter(actions[i],measurement[i])
         print("Iteration",i+1,"done\n")
   
if __name__ == "__main__":
    main()