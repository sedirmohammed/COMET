
class CleaningSchedule:

    def __init__(self):
        self.cleaning_settings = []

    def add_cleaning_setting(self, cleaning_setting):
        self.cleaning_settings.append(cleaning_setting)

    def get_cleaning_settings_for_feature(self, feature):
        collected_cleaning_settings = []
        for cleaning_setting in self.cleaning_settings:
            if cleaning_setting.feature == feature:
                collected_cleaning_settings.append(cleaning_setting)
        return collected_cleaning_settings

    def get_real_pollution_levels_for_feature(self, feature):
        collected_pollution_levels = []
        for cleaning_setting in self.cleaning_settings:
            if cleaning_setting.feature == feature:
                collected_pollution_levels.append(cleaning_setting.real_pollution_level)

    def get_cleaning_settings(self):
        return self.cleaning_settings


class CleaningSetting:

    def __init__(self):
        self.feature = None
        self.reg_pollution_level = -1.0
        self.reg_predicted_f1 = -1.0
        self.real_f1 = -1.0
        self.used_budget = -1.0
        self.f1_gain_predicted = -1.0
        self.assumed_pollution_level = -1.0
        self.real_pollution_level = -1.0

    def set_feature(self, feature):
        self.feature = feature

    def set_reg_pollution_level(self, reg_pollution_level):
        self.reg_pollution_level = reg_pollution_level

    def set_reg_predicted_f1(self, reg_predicted_f1):
        self.reg_predicted_f1 = reg_predicted_f1

    def set_real_f1(self, real_f1):
        self.real_f1 = real_f1

    def set_used_budget(self, used_budget):
        self.used_budget = used_budget

    def set_f1_gain_predicted(self, f1_gain_predicted):
        self.f1_gain_predicted = f1_gain_predicted

    def set_assumed_pollution_level(self, assumed_pollution_level):
        self.assumed_pollution_level = assumed_pollution_level

    def set_real_pollution_level(self, real_pollution_level):
        self.real_pollution_level = real_pollution_level

    def get_feature(self):
        return self.feature

    def get_reg_pollution_level(self):
        return self.reg_pollution_level

    def get_reg_predicted_f1(self):
        return self.reg_predicted_f1

    def get_real_f1(self):
        return self.real_f1

    def get_used_budget(self):
        return self.used_budget

    def get_f1_gain_predicted(self):
        return self.f1_gain_predicted

    def get_assumed_pollution_level(self):
        return self.assumed_pollution_level

    def get_real_pollution_level(self):
        return self.real_pollution_level
