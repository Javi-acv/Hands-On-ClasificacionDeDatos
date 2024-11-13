class SimpleLinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.beta_0 = None
        self.beta_1 = None

    def calculate_parameters(self):
        x_mean = sum(self.x) / len(self.x)
        y_mean = sum(self.y) / len(self.y)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(self.x, self.y))
        denominator = sum((xi - x_mean) ** 2 for xi in self.x)
        self.beta_1 = numerator / denominator
        
        self.beta_0 = y_mean - self.beta_1 * x_mean

    def predict(self, x):
        return self.beta_0 + self.beta_1 * x

    def get_parameters(self):
        return self.beta_0, self.beta_1


advertising_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sales_data = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

model = SimpleLinearRegression(advertising_data, sales_data)
model.calculate_parameters()

beta_0, beta_1 = model.get_parameters()
print(f"Pendiente (Beta_1): {beta_1}")
print(f"Intercepto (Beta_0): {beta_0}")


prediccion = model.predict(60)
print(f"Predicción de ventas para Advertising = 60 millones: {prediccion} millones")

