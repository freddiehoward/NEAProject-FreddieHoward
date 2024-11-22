def create_batches(X, y, batch_size):
    '''
    Function to manually split the dataset into batches of input (X) and output (y).
    - Parameters:
        X: The input data to be split into batches.
        y: The output data corresponding to the inputs.
        batch_size: The number of samples in each batch.
    - Returns:
        A list of tuples, where each tuple contains a batch of input and output data.
    '''
    batches = []
    for i in range(0, len(X+1), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        if len(X_batch) == batch_size:  # Ensure only full batches are included
            batches.append((X_batch, y_batch))
    return batches
