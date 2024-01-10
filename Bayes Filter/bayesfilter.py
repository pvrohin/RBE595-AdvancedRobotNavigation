#Initial state of the door is unknown
bel_x0_open = 0.5
bel_x0_closed = 0.5

#Intitialize the measurement model
p_zt_open_xt_open = 0.6
p_zt_open_xt_closed = 0.4
p_zt_closed_xt_open = 0.2
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

#Write the Bayes Filter function
def bayes_filter(action,measurement):
    global bel_x0_open,bel_x0_closed
    if(action=="do_nothing" and measurement=="open"):
        bel_bar_x1_open = (p_xt_open_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_open_ut_do_nothing_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_closed_ut_do_nothing_xt_1_closed*bel_x0_closed)

        bel_x1_open = p_zt_open_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_open_xt_closed*bel_bar_x1_closed

        norm = 1/(bel_x1_open+bel_x1_closed)

        bel_x1_open = norm*bel_x1_open

        bel_x1_closed = norm*bel_x1_closed

        bel_x0_open = bel_x1_open
        bel_x0_closed = bel_x1_closed

        print("Probability the door is open is ",bel_x0_open)
        print("Probability the door is closed is ",bel_x0_closed)

    if(action=="open" and measurement=="open"):
        bel_bar_x1_open = (p_xt_open_ut_push_xt_1_open*bel_x0_open) + (p_xt_open_ut_push_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_push_xt_1_open*bel_x0_open) + (p_xt_closed_ut_push_xt_1_closed*bel_x0_closed)

        bel_x1_open = p_zt_open_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_open_xt_closed*bel_bar_x1_closed

        norm = 1/(bel_x1_open+bel_x1_closed)

        bel_x1_open = norm*bel_x1_open

        bel_x1_closed = norm*bel_x1_closed

        bel_x0_open = bel_x1_open
        bel_x0_closed = bel_x1_closed

        print("Probability the door is open is ",bel_x0_open)
        print("Probability the door is closed is ",bel_x0_closed)

    if(action=="open" and measurement=="closed"):
        bel_bar_x1_open = (p_xt_open_ut_push_xt_1_open*bel_x0_open) + (p_xt_open_ut_push_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_push_xt_1_open*bel_x0_open) + (p_xt_closed_ut_push_xt_1_closed*bel_x0_closed)

        bel_x1_open = p_zt_closed_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_closed_xt_closed*bel_bar_x1_closed

        norm = 1/(bel_x1_open+bel_x1_closed)

        bel_x1_open = norm*bel_x1_open

        bel_x1_closed = norm*bel_x1_closed

        bel_x0_open = bel_x1_open
        bel_x0_closed = bel_x1_closed

        print("Probability the door is open is ",bel_x0_open)
        print("Probability the door is closed is ",bel_x0_closed)

    if(action=="do_nothing" and measurement=="closed"):
        bel_bar_x1_open = (p_xt_open_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_open_ut_do_nothing_xt_1_closed*bel_x0_closed)

        bel_bar_x1_closed = (p_xt_closed_ut_do_nothing_xt_1_open*bel_x0_open) + (p_xt_closed_ut_do_nothing_xt_1_closed*bel_x0_closed)

        bel_x1_open = p_zt_closed_xt_open*bel_bar_x1_open

        bel_x1_closed = p_zt_closed_xt_closed*bel_bar_x1_closed

        norm = 1/(bel_x1_open+bel_x1_closed)

        bel_x1_open = norm*bel_x1_open

        bel_x1_closed = norm*bel_x1_closed

        bel_x0_open = bel_x1_open
        bel_x0_closed = bel_x1_closed

        print("Probability the door is open is ",bel_x0_open)
        print("Probability the door is closed is ",bel_x0_closed)

def main():
    actions = ["do_nothing","open","do_nothing","open","do_nothing"]
    measurement = ["closed","closed","closed","open","open"]

    for i in range(0,5):
         print("Action",actions[i])
         print("Measurement",measurement[i])
         bayes_filter(actions[i],measurement[i])
         print("Iteration",i,"done\n")
   
if __name__ == "__main__":
    main()