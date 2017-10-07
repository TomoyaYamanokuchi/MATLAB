classdef temp_class(self)
    properties
       self.x = 2
       self.y = 3
    end
    methods
        function self = temp_class()
            self.x = 3
            self.y = 5
        end
        
        function out = get_xy(self)
            self.x = 4
            self.y = 
            out = self.x*self.y
        end
        
    end
end
    