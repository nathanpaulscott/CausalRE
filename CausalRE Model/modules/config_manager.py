import copy




class ImmutableNamespace:
    def __init__(self, **kwargs):
        # Store attributes in a private dictionary to prevent direct access and modification
        self.__dict__['data'] = kwargs

    def __getattr__(self, name):
        # Allow attribute access via self.name
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Prevent modifications directly via attribute setting
        raise AttributeError("Cannot modify attributes directly. Use the update method.")

    def update(self, updates):
        # Allow updates via this internal method
        self.data.update(updates)


    def get_data_copy(self):
        #Return a dict of the params as a deep copy of the data to ensure immutability is maintained
        return copy.deepcopy(self.__dict__['data'])



class Config:
    """
    Config class that encapsulates configuration settings.
    It provides controlled access and modification of these settings.
    """
    def __init__(self, initial_settings):
        # Initialize the configuration with an immutable namespace
        self.cfg = ImmutableNamespace(**initial_settings)

    def update(self, updates):
        """
        Updates the configuration settings in a controlled manner through the ImmutableNamespace's update method.
        Args: updates (dict): A dictionary of updates to apply to the configuration.
        """
        # Delegate the update call to the ImmutableNamespace's update method
        self.cfg.update(updates)

    @property
    def as_namespace(self):
        """
        Provides access to the configuration settings via the namespace syntax but ensures immutability.
        Returns:  ImmutableNamespace: The configuration settings accessible via namespace syntax.
        """
        return self.cfg






'''
class Config_old:
    #Makes the config oject which has a property .cfg which is a namespace with all the relevant config parameters
    def __init__(self, initial_settings):
        self.cfg = SimpleNamespace(**initial_settings)

    def update(self, updates):
        #Update the internal config namespace with additional or existing parameters
        #input is a dict
        for key, value in updates.items():
            setattr(self.cfg, key, value)

    def as_namespace(self):
        # Return settings as a SimpleNamespace (or similar), if needed
        return self.cfg

'''