import time


class NoProgressShower:

    def __init__(self):
        pass

    def update_post(self, new_post, step=None):
        pass

    def update(self):
        pass

    def terminate(self):
        pass

    @property
    def percentage(self):
        return None


class ProgressShower(NoProgressShower):

    def __init__(self, total_steps=100, pre='', post='',
                 show_percentage=True, show_bar=True, show_etr=True
                 , empty='_', fill='█'):
        """
        Simple progress shower. Consists of a
        Parameters
        ----------
        total_steps: Number or iterable
            Total number of steps. If iterable, its length is used
        pre: str
            The string to add at the start of the progress bar
        post: str
            The string to add at the end of the progress bar

        show_percentage: bool
            Show completed percentage
        show_bar: bool
            Show a bar with completed percentage
        show_etr: bool
            Show the expected remaining time.

        empty: str
            The character to indicated the remaining progress. Ignored for show_bar=False
        fill: str
            The character to indicate the progress. Ignored for show_bar=False

        Notes
        -----
        Expected remaining time is computed using the passed execution time, using the average of the passed steps
        """

        super().__init__()

        try:
            total_steps = int(total_steps)
        except (ValueError, TypeError):
            total_steps = len(total_steps)

        assert total_steps >= 0
        self.total_steps = total_steps
        self.current = 0

        assert isinstance(empty, str) and len(empty) == 1, f'invalid value for empty: {empty}'
        assert isinstance(fill, str) and len(fill) == 1, f'invalid value for fill: {fill}'
        self.empty = empty
        self.fill = fill

        self.pre = '' if pre == '' else str(pre) + ' '

        self.show_percentage = show_percentage
        self.show_bar = show_bar

        self.start_time = time.process_time()
        self.show_eta = show_etr

        self.finished = False

        self.__post = None
        self.__eta_process_time = None
        self.update_post(post)

    def update_post(self, new_post, step=None):
        self.__post = '' if new_post == '' else ' ' + str(new_post)
        if step is not None:
            self.update(step)
        else:
            self.__draw()

    def update(self, step=1):
        """
        Update the progress.

        Terminates if the new value pass self.total_steps.

        Parameters
        ----------
        step: Number
            Number of steps to take
        """
        if self.finished:
            return

        assert step >= 0
        self.current += step

        if self.show_eta and self.current > 0:
            time_that_has_passed = time.process_time() - self.start_time
            self.__eta_process_time = self.start_time + time_that_has_passed / self.current * self.total_steps

        if self.current >= self.total_steps:
            self.terminate()
        else:
            self.__draw()

    def __draw(self):
        """
        Shows the output
        """
        bar = self.fill * self.percentage + self.empty * (100 - self.percentage) if self.show_bar else ''

        if self.show_percentage or self.show_eta:
            percentage = f' ['
            percentage += f"{self.percentage:02}%" if self.show_percentage else ""

            if self.show_percentage and self.show_eta:
                percentage += ' / '
            if self.__eta_process_time is not None:
                rem = self.__eta_process_time - time.process_time()
                percentage += f'{int(rem // 60):02}:{int(rem % 60):02}'
            percentage += ']'
        else:
            percentage = ''

        print(f'\r{self.pre}{bar}{percentage}{self.__post}', end='', flush=True)

    def terminate(self):
        """
        Terminates the Progress Shower instance.

        Adds new line and prevents further drawing

        """
        self.update_post('')
        self.current = self.total_steps
        self.__draw()
        self.finished = True
        print()

    @property
    def percentage(self):
        return int(self.current / self.total_steps * 100)
