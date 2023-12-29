
# Import the function of task A
from A.task_A import run_A

# Import the function of task B
from B.task_B import run_B

def main():
    # Get the result of task A
    print("Result from task A:")
    run_A()
    
    # Get the result of task B
    print("Result from task B:")
    run_B()
    
if __name__ == "__main__":
    main()
