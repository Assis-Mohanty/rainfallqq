from terranova.logging import logger
import sys

class NetworkSecurityException(Exception):
    def __init__(self,error_message,error_details:sys):
        self.error_message=error_message
        _,_,exc_tb=error_details.exc_info()
        self.lineno=exc_tb.tb_lineno
        self.file_name=exc_tb.tb_frame.f_code.co_filename
    
    def __str__(self):
        return "\n Error occured in python script name \n [{0}] \n line number [{1}] \n error message \n [{2}] ".format(
            self.file_name,self.lineno,str(self.error_message)
        )

# if __name__=='__main__':
#     try:
#         logger.logging.info("enter the try block")
#         a=1/0
#         print("qqq",a)
#     except Exception as e:
#         raise NetworkSecurityException(e,sys)
        