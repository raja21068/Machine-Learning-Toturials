try:
    x = 1
    y = 1
    print ("Result of x/y: ", x / y)
except (ZeroDivisionError):
    print("Can not divide by zero")
except (TypeError):
    print("Wrong data type, division is allowed on numeric data type only")
except:
    #print ("Unexpected error occurred", '\n', "Error Type: ", sys.exc_info() [0], '\n', "Error Msg: ", sys.exc_info()[1])

    # Below code will open a file and try to convert the content to integer
    try:
        f = open('vechicles.txt')
        print (f.readline())
       # i = int(s.strip())
    except IOError as e:
        print( "I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
     print ("Could not convert data to an integer.")
   # except:
    # print("Unexpected error occurred", '\n', "Error Type: ", sys.exc_info())[0], '\n', "Error Msg: ", sys.exc_info()[1])
    finally:
        f.close()
        print("file has been closed")