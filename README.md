# IDL_22
Repository used for IDL assigments for group 22 at Leiden Uni
# Assignment 1 Part 1
## Running the Experiments

1.  Navigate to the project directory containing the experiment scripts (the "experiments folder").
2.  Install all the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Ensure that the `manager_N.py` script you wish to run is in the same directory as its corresponding `worker_N.py`.
4.  To run experiment `N`, execute the following command in your terminal:

    ```bash
    # Replace 'N' with the experiment number (e.g., 1, 2, 3)
    python manager_N.py
    ```
