class OptimizationConfig(object):
    """docstring for OptimizationConfig"""
    def __init__(self):
        super(OptimizationConfig, self).__init__()
        self.dataset = '20ng'
        self.model = 'gcn'  # 'gcn', 'dense'
        self.learning_rate = 0.02  # Initial learning rate.
        self.epochs = 200  # Number of epochs to train.
        self.hidden1 = 200  # Number of units in hidden layer 1.
        self.dropout = 0.5  # Dropout rate (1 - keep probability).
        self.early_stopping = 10  # Tolerance for early stopping (# of epochs).
