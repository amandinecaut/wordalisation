from abc import ABC, abstractmethod
import textwrap
class ExplanationProvider(ABC):
    """ Base interface for generation hover explanations. """
    @abstractmethod
    def get_explanation(self, metric_name: str, entity, value: float) -> str:
        """ Rturns a short string to incluse in hover text"""
        pass
    def wrap_text(self, text, width=85):
        """ Utility method to wrap text for better readability. """
        return "<br>".join(textwrap.wrap(text, width=width))
    
class CountryExplanationProvider(ExplanationProvider):
    """ Explanation provider for country-level metrics. """
    def __init__(self, relevant_questions, drill_down_metrics):
        self.relevant_questions = relevant_questions
        self.drill_down_metrics = drill_down_metrics
    def get_explanation(self, metric_name, entity, value):
        explanation="<br>"
        # remove _Z from metric_name
        metric_name = metric_name.replace("_Z", "")
        # metric_key_lower=metric_name.lower()
        
        if metric_name.lower() in self.drill_down_metrics:
            if entity[metric_name + "_Z"] > 0:
                index = 1
            else:
                index = 0
            question, value = self.drill_down_metrics[metric_name.lower()]
            question, value = question[index], value[index]
            explanation += "In response to the question '"
            explanation += self.relevant_questions[metric_name][question][0]
            explanation += "', on average participants "
            explanation += self.relevant_questions[metric_name][question][1]
            explanation += " '"
            explanation += self.relevant_questions[metric_name][question][2][str(value)]
            explanation += "' "
            explanation += self.relevant_questions[metric_name][question][3]
            explanation += ". "
        elif metric_name in self.drill_down_metrics:
            if entity[metric_name + "_Z"] > 0:
                index = 1
            else:
                index = 0

            question, value = self.drill_down_metrics[metric_name]
            question, value = question[index], value[index]
            explanation += "In response to the question '"
            explanation += self.relevant_questions[metric_name][question][0]
            explanation += "', on average participants "
            explanation += self.relevant_questions[metric_name][question][1]
            explanation += " '"
            explanation += self.relevant_questions[metric_name][question][2][str(value)]
            explanation += "' "
            explanation += self.relevant_questions[metric_name][question][3]
            explanation += ". "

        return self.wrap_text(explanation)
         
class PersonExplanationProvider(ExplanationProvider):
      def __init__(self, questions):
          self.questions = questions
          # Define where each trait’s questions lie in person_metrics
          self.metric_ranges = {
            "Extraversion": (0, 10),
            "Neuroticism": (10, 20),
            "Agreeableness": (20, 30),
            "Conscientiousness": (30, 40),
            "Openness": (40, 50),
        }
      def get_explanation(self, metric_name, entity, value):       
        explanation = "<br>"
        metric_name = metric_name.replace("_Z", "")
        start, end = self.metric_ranges[metric_name.lower().capitalize()]
        sub_metrics = entity[start:end]
        z_value = entity[metric_name+"_Z"]
        if abs(z_value) >1 :
            if z_value >0:
                index_max= sub_metrics.idxmax()
                explanation= f"They said that {self.questions[index_max][0]}."
            else:
                index_min= sub_metrics.idxmin()
                explanation= f"They said that {self.questions[index_min][0]}."
        return self.wrap_text(explanation)
          