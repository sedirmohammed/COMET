from .cleaning_strategy import CleaningStrategy


class DataCleaning:

    def __init__(self, strategy: CleaningStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: CleaningStrategy):
        self._strategy = strategy

    def perform_cleaning(self, *args, **kwargs):
        return self._strategy.select_cleaning_setting(*args, **kwargs)

    def get_cleaning_buffer(self):
        return self._strategy.get_cleaning_buffer()
