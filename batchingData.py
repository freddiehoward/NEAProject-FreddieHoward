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

'''
Define the batch size and create batches from the training data.
'''
batch_size = 64
train_batches = create_batches(X_train, y_train, batch_size)

'''
Print the number of batches and the shapes of the first batch to verify correctness.
'''
print(f"Number of training batches: {len(train_batches)}")
first_batch_X, first_batch_y = train_batches[0]
print(f"First Batch Input Shape: {first_batch_X.shape}")
print(f"First Batch Output Shape: {first_batch_y.shape}")