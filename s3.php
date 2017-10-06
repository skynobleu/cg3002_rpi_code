<?php

// Include the SDK using the Composer autoloader
// date_default_timezone_set('Asia/Singapore');
require '/home/pi/vendor/autoload.php';
use Aws\S3\S3Client;
use Aws\S3\Exception\S3Exception;
use Aws\S3\Exception\NoSuchBucketException;
use Aws\Exception\AwsException;
use Aws\Credentials\Credentials;

/**
 * Class s3_handler
 *
 * Handles the CRUD of resources in the designated S3 bucket
 * Created with reference to http://docs.aws.amazon.com/aws-sdk-php/v2/guide/service-s3.html
 *
 * ethan
 *
 */

class s3_handler {
    //AWS Service Access Credentials
    private $access_id= 'AKIAIUEVUE3BIK63ZA2Q';
    private $secret_access_key = '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';
    private $bucket = "cg3002-group5";
    private $region = 'ap-southeast-1';

    private $credentials;
    private $client;

    //Initialises the S3 PHP Client.
    function __construct() {
        $this->credentials = new Credentials($this->access_id, $this->secret_access_key);
        $this->client = S3Client::factory(array(
            'credentials' => $this->credentials,
            'version' => 'latest',
            'region'  => $this->region
        ));

        //enable the storage and retrieval of data using php built-in functions
        //required for function getFileFromS3
        $this->client->registerStreamWrapper();
    }


    /**
     * Uploads specified directory into virtual folder on S3 bucket
     *
     * @params $directory, $folder
     *
     * all path/directory information in the bucket must be included in $folder.
     * e.g 'mainfolder/innerfolder/'
     *
     * @return true if success
     */

    function uploadDirectoryToS3($directory, $folder){
        //uploads into bucket under virtual folder specified as keyPrefix
        try {
            $this->client->uploadDirectory($directory, $this->bucket, $folder, array(
                'params' => array('ACL' => 'public-read'),
                'concurrency' => 20,
                'debug' => true
            ));
        } catch(NoSuchBucketExceptionException $e) {
            echo $e->getMessage();
            $return['error'] = "no such bucket exists, please check bucket name";

        } catch(Exception $e){
            echo $e->getMessage();
            $return['error'] = $e->getMessage();

        }

        $return['success'] = true;
        return $return;
    }

    /**
     * Lists all objects within the S3 bucket
     *
     */
    function listObjectsInBucket(){
        $iterator = $this->client->getIterator('ListObjects', array(
            'Bucket' => $this->bucket
        ));

        foreach ($iterator as $object) {
            echo $object['Key'] . "\n";
        }
    }
    function findObject($prefix){
        $iterator = $this->client->getIterator('ListObjects', array(
            'Bucket' => $this->bucket,
            'Prefix' => $prefix
        ));
        $array = array_slice(iterator_to_array($iterator), 1);
        //print_r($array);

        foreach($array as &$row){
            //echo $row['Key'];
            $row['url'] = $this->client->getObjectUrl($this->bucket, $row['Key']);
        }
        print_r($array);
        return $array;
    }
    /**
     * Finds object of $filename within the S3 bucket
     *
     * push out the key/s to the file when found within the bucket
     * @param $filename
     * (if found) false (if not found)
     * @return array
     */
    function findFileInS3($filename) {
        $iterator = $this->client->getIterator('ListObjects', array(
            'Bucket' => $this->bucket
        ));
        $return = array();
        //loop through every object
        foreach ($iterator as $object) {
            //obtain file name of object
            $paths = explode('/', $object['Key']);

            //if filename matches
            if( end($paths) == $filename){
                array_push($return, $object['key']);
            }
        }

        return $return;
    }

    /**
     * Upload file from specified filepath to folder(location) to S3 bucket
     *
     * @params $pathToFile, $filename, string $folder
     */

    function uploadFileToS3($pathToFile, $filename, $folder = false) {
        if ($folder){
            $filename = "$folder/".$filename;
        }
        try {
            $result = $this->client->putObject(array(
                'Bucket' => $this->bucket,
                'Key' => $filename,
                'SourceFile' => $pathToFile,

            ));

            $this->client->waitUntil('ObjectExists', array(
                'Bucket' => $this->bucket,
                'Key'    => $filename
            ));
        } catch (NoSuchBucketException $e) {
            $return["error"] = $e->getMessage();
            die($return["error"]);
            return $return;
        }
        //echo "$filename uploaded to $this->bucket";

        $return['success'] = true;

        return $return;

    }

    /**
     * Obtains specified file from S3 and stores it in a variable that is returned
     *
     * @params $filename, string $folder (key prefix)
     * @return false (if object is not found)| $object (if found)
     */

    //obtains specified file from bucket from (folder) if any
    function getFileFromS3($filename, $folder = false){
        if($folder){
            $filename = "$folder/".$filename;
        }
        $exists = $this->client->doesObjectExist($this->bucket, $filename);
        if (!$exists) {
            $return["error"] = "object does not exists";
            return $return;
        }
        try {
            $this->client->registerStreamWrapper();
            $obj = file_get_contents("s3://{$this->bucket}/{$filename}");
            return $obj;
        } catch (NoSuchBucketException $e){
            $return['error'] = $e->getMessage();
            return $return;
        }
    }

    /**
     * Downloads specified file from bucket to local directory
     *
     * @params: $directory (path on disc), $filename, optional: $folder (key prefix)
     * e.g $directory = 'local/folder/', saved file can be accessed at 'local/folder/<$filename>'
     */
    function downloadFileFromS3($directory, $filename, $folder = ''){

        if($folder){

            $filename = "$folder/".$filename;
        }

        $exists = $this->client->doesObjectExist($this->bucket, $filename);
        if (!$exists) {
            $return["error"] = "does not exist";
            return $exists;
        }
        try {
            $result = $this->client->getObject(array(
                'Bucket' => $this->bucket,
                'Key' => $filename,
                'SaveAs' => $directory
            ));
        } catch (NoSuchBucketException $e){
            $return['error'] = "bucket does not exist, please check bucket name";
            return $return;
        }
        $return['success'] = true;
        return $return;
    }

    /**
     *  Generate accessible url if object is available
     *
     * Valid URLS:
     * http(s)://<bucket>.s3.amazonaws.com/<object>
     * http(s)://s3.amazonaws.com/<bucket>/<object>
     *
     * @params: $filename, optional: $folder (key prefix)
     */
    function generateObjectUrl($filename, $folder = false){
        if($folder){
            $filename = "$folder/".$filename;
        }

        $exists = $this->client->doesObjectExist($this->bucket, $filename);
        if (!$exists) {
            $return['error'] = "object does not exist";
            return $return;
        }

        try {
            $url = $this->client->getObjectUrl($this->bucket, $filename, '30 minutes');
            return $url;
        } catch (NoSuchBucketException $e) {
            $return['error'] = $e->getMessage();
            return $return;
        }
        $return['error'] = "an error has occured";
    }

    /**
     * Generates base url pointing to the default directory of the s3 bucket
     *
     * format:
     * http(s)://<bucket>.s3.amazonaws.com/<folder/>
     *
     * @params  $folder (can also be a directory: folder/innerfolder )
     */
    function generateDirectoryUrl($folder = false) {
        //check if bucket exists

        $exist = $this->client->doesBucketExist($this->bucket);
        if(!$exist) {
            $return['error'] = "bucket does not exist";
            return $return;
        }
        $url = "http://".$this->bucket.".s3.amazonaws.com/";
        if ($folder) {
            $url = $url.$folder."/";
        }
        return $url;
    }
    /**
    * Generates a copy of a new file in the same folder.
    *
    * @params $filename (sourcefile), $copyname (name of copy), $folder (directory)
    */
   function cloneObject($filename, $copyname, $folder = false){
       if($folder){
           $filename = "$folder/".$filename;
           $copyname = "$folder/".$copyname;
       }

       try {
           $this->client->copyObject(array(
               'Bucket' => $this->bucket,
               'Key' => "{$copyname}",
               'CopySource' => "{$this->bucket}/{$filename}",
           ));
       } catch (NoSuchBucketException $e){
           $return['error'] = $e->getMessage();

       } catch (Exception $e) {
           $return['error'] = $e->getTraceAsString();
       }
   }

}
$s3 = new s3_handler();
$s3->uploadDirectoryToS3('bin/log', 'log');




?>
