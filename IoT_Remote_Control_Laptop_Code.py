import paho.mqtt.client as mqtt
import logging
import time



# Set up logging
logging.basicConfig(level=logging.DEBUG)


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("Connected successfully")
       
        client.subscribe("output_topic")
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    print(f"Received message: {msg.payload.decode()} on topic {msg.topic}")

client = mqtt.Client(protocol=mqtt.MQTTv5, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set("<user name>", "<password>")

try:
    print("Attempting to connect...")
    client.connect(",IP adress>", <tcp port>, 60)
    client.loop_start()
    
    # Wait for connection to establish
    time.sleep(2)
    
    print("Attempting to publish...")
    while True:
        input_data = input("Enter input commands w,a,s,d,and f: ")
        client.publish("output_topic", input_data, qos=1)
        if input_data == "exit":
            break
        
    result = client.publish("output_topic", "Your output data here", qos=1)
    
    # Wait for message to be published
    #time.sleep(2)
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("Message published successfully")
    else:
        print(f"Failed to publish message. Error code: {result.rc}")
    
    client.loop_stop()
    client.disconnect()
except Exception as e:
    print(f"Connection failed: {e}")
finally:
    print("Execution completed")

