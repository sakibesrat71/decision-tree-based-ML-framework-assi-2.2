class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y):
        data = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"{name} must implement transform()")

            if hasattr(step, "fit_transform"):
                data = step.fit_transform(data)
            else:
                step.fit(data)
                data = step.transform(data)

        model_name, model = self.steps[-1]

        if not hasattr(model, "predict"):
            raise TypeError(f"{model_name} must implement predict()")

        model.fit(data, y)

    def predict(self, X):
        data = X

        for _, step in self.steps[:-1]:
            data = step.transform(data)

        return self.steps[-1][1].predict(data)