







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