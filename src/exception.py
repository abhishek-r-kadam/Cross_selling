import sys


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "The Error is in the File :{0} at the line number :{1}, error is :{2}".format(file_name,exc_tb.tb_lineno,str(error))
    
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_messsage= error_message_detail(error_message,error_detail=error_detail)
        
        
    def __str__(self) -> str:
        return self.error_messsage
        