import six

from .terminalsize import get_terminal_size


def print_progress_bar(iteration,
                       total,
                       prefix='',
                       suffix='',
                       decimals=1,
                       length=100,
                       fill='█'):
    """
    print progress bar that fill the width of terminal. (omit prefix if it is too long)

    :param iteration: Required : current iteration (Int)
    :param total: Required  : total iterations (Int)
    :param prefix: Optional  : prefix string (Str)
    :param suffix: Optional  : suffix string (Str)
    :param decimals: Optional  : positive number of decimals in percent complete (Int)
    :param length: Optional  : character length of bar (Int)
    :param fill: Optional  : bar fill character (Str)
    :return: None
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    width, _ = get_terminal_size()
    print_length = len('%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    if print_length > width:
        ignored_length = print_length + 10 - width
        prefix = prefix[:(len(prefix) - ignored_length) // 2] + '....' + prefix[(len(prefix) + ignored_length) // 2:]

    # clear print line before output
    six.print_(' ' * width, end='\r')

    six.print_('%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
