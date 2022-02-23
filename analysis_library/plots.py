class BarPlot:

    def __call__(self, data, dependent, independent):
        f = data.bars(x=independent, y=dependent)
        return {'barplot': f}
