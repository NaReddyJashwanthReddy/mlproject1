#used for manipulating diff. parts of python runtime environment.here we are using it for exception handling information
import sys

def error_message_details(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # 1 and 2 are not important 3rd var is import because it consist the details of where and which error has occure
    file_name=exc_tb.tb_frame.f_code.co_filename 
    error_message="Error occured in python script name[{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message
    
