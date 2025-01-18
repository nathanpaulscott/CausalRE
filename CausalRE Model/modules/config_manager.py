import copy, json
from json import JSONEncoder
import datetime
from pathlib import Path

class CustomEncoder(JSONEncoder):
    def default(self, obj):
        # Convert datetime objects to strings
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        # Convert pathlib.Path objects (including WindowsPath and PosixPath) to strings
        elif isinstance(obj, Path):
            return str(obj)
        # Convert any other non-serializable objects to their string representation
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, obj)        
        
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



    @property
    def dump_as_json_str(self):
        ignore = ['all_span_ids', 'logger', 'torch_precision']
        data_copy = self.cfg.get_data_copy()
        for key in ignore:
            data_copy.pop(key, None)

        json_string = json.dumps(data_copy, indent=4, sort_keys=True, cls=CustomEncoder)
        # Attempt to align values
        lines = json_string.splitlines()
        longest_key_length = max((len(line.split(':')[0]) for line in lines if ':' in line), default=0)
        aligned_json = []
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                # Adjust the key length to the longest key + 1 to account for the colon
                new_line = key.ljust(longest_key_length + 1) + ':' + value
                aligned_json.append(new_line)
            else:
                aligned_json.append(line)
        return '\n'.join(aligned_json)
