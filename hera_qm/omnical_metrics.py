
def get_omnical_metrics_dict():
    """ Simple function that returns dictionary with metric names as keys and
    their descriptions as values. This is used by hera_mc to populate the table
    of metrics and their descriptions.

    Returns:
    metrics_dict -- Dictionary with metric names as keys and descriptions as values.
    """
    metrics_dict = {'omnical_quality': 'Quality of cal solution (chi-squared) '
                    'for each antenna.',
                    'omnical_total_quality': 'Quality of overall cal solution '
                    '(chi-squared) across entire array.'}
    return metrics_dict
