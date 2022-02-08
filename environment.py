class Environment:
    PRICE_IDX = 4
    ####
    STEP = 1
    ####

    def __init__(self, chart_data=None, training_data=None):
        self.chart_data = chart_data
        self.training_data = training_data
        self.observation = None
        self.idx = -1

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        if len(self.chart_data) > self.idx + self.STEP:
            self.idx += self.STEP
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def get_next_price(self):
        if len(self.chart_data) > (self.idx + self.STEP):
            return self.chart_data.iloc[self.idx + self.STEP][self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data

    def set_training_data(self, training_data):
        self.training_data = training_data

    def get_training_data_shape(self):
        return self.training_data.shape[1]