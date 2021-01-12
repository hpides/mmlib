from colorama import Fore, Style


def _print_info(message):
    print(Fore.GREEN + "INFO: " + message + Style.RESET_ALL)


def _print_warning(message):
    print(Fore.YELLOW + "WARNING: " + message + Style.RESET_ALL)
