#include "ns3/core-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/opengym-module.h"
//#include "ns3/csma-module.h"
#include "ns3/propagation-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-flow-classifier.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/inet-socket-address.h"
#include "ns3/inet6-socket-address.h"
#include "ns3/header.h"
#include "ns3/ipv4-header.h"
#include "ns3/ipv4-address.h"
#include "ns3/ipv4-l3-protocol.h"

#include <fstream>
#include <string>
#include <math.h>
#include <ctime>   //timestampi
#include <iomanip> // put_time
#include <vector>
#include <deque>
#include <algorithm>
#include <csignal>
#include "scenario.h"


using namespace std;
using namespace ns3;

NS_LOG_COMPONENT_DEFINE("OpenGym");

void installTrafficGenerator(Ptr<ns3::Node> fromNode, Ptr<ns3::Node> toNode, int port, string offeredLoad, double startTime);
void PopulateARPcache();
void recordHistory();

double envStepTime = 0.1;
double simulationTime = 10; //seconds
double current_time = 0.0;
bool verbose = false;
int end_delay = 0;
bool dry_run = false;

Ptr<FlowMonitor> monitor;
FlowMonitorHelper flowmon;
ofstream outfile ("scratch/RLinWiFi-Decentralized-v02/CW_data.csv", fstream::out);

const int MAX_WIFI = 50;
//uint32_t CW = 0;
uint32_t nWifi = 2;
std::vector<uint32_t> CW(MAX_WIFI, 0);

// para poder inicializar com zero declarei desse jeito
std::vector<uint64_t> g_rxPktNum(MAX_WIFI, 0);
std::vector<uint64_t> g_txPktNum(MAX_WIFI, 0);

// Our data structure for the scenario needs to be a vector of [history_length, 2]
uint32_t history_length = 20;

//deque<float> history;
//std::vector<std::deque<float>> history(80); //funcionando adicionado por sheila
std::vector<std::deque<float>> history(MAX_WIFI);

string type = "discrete";
bool non_zero_start = false;
Scenario *wifiScenario;


/*
Define observation space
*/
Ptr<OpenGymSpace> MyGetObservationSpace(void)
{
    float low = 0.0;
    float high =10.0;
    std::vector<uint32_t> shape = {
        history_length,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetObservationSpace: " << space);
    //std::cout << "Size: " << space.GetShape() << std::endl;
    return space;
}

/*
Define action space
*/
Ptr<OpenGymSpace> MyGetActionSpace(void)
{
    float low = 0.0;
    float high = 10.0;
    std::vector<uint32_t> shape = {
        1,
    };
    std::string dtype = TypeNameGet<float>();
    Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace>(low, high, shape, dtype);
    if (verbose)
        NS_LOG_UNCOND("MyGetActionSpace: " << space);
    return space;
}

/*
Define extra info. Optional
*/
//uint64_t g_rxPktNum = 0;
//uint64_t g_txPktNum = 0;

/*std::vector<uint64_t> g_rxPktNum = {
        2,
};
std::vector<uint64_t> g_txPktNum = {
        2,
};*/

double jain_index(void)
{
    double flowThr;
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
    std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();

    double nominator;
    double denominator;
    double n=0;
    double station_id = 0;
    for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
    {
        flowThr = i->second.rxBytes;
        flowThr /= wifiScenario->getStationUptime(station_id, current_time);
        if(flowThr>0){
            nominator += flowThr;
            denominator += flowThr*flowThr;
            n++;
        }
        station_id++;
    }
    nominator *= nominator;
    denominator *= n;
    return nominator/denominator;
}

std::string MyGetExtraInfo(void)
{
    static float ticks = 0.0;
    static float lastValue[MAX_WIFI];
    ticks += envStepTime;
    std::string myInfo;  
    float sentMbytes = 0;  
    
    for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++)
    {
         float obs = g_rxPktNum[sta_id] - lastValue[sta_id];
         lastValue[sta_id] = g_rxPktNum[sta_id];         

         sentMbytes += obs * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;
    }

    myInfo = std::to_string(sentMbytes);
	
	for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++){
		myInfo += "|" + to_string(CW[sta_id]);
	}
    myInfo = myInfo + "|" + to_string(wifiScenario->getActiveStationCount(ticks)) + "|";
    myInfo = myInfo + to_string(jain_index());
    
    if (verbose)
        NS_LOG_UNCOND("MyGetExtraInfo: " << myInfo);

    return myInfo;
}

/*
Execute received actions
*/
bool MyExecuteActions(Ptr<OpenGymDataContainer> action)
{
   if (verbose)
        NS_LOG_UNCOND("MyExecuteActions: " << action);
        std::cout << "MyExecuteActions: " << action << std::endl;
       

    Ptr<OpenGymBoxContainer<float>> box = DynamicCast<OpenGymBoxContainer<float>>(action);
    std::vector<float> actionVector = box->GetData();
   	uint32_t min_cw = 16;
	uint32_t max_cw = 1024;	
	
	for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++) //para percorrer/acessar as Stations
    {
		if (type == "discrete")
		{
			CW[sta_id] = pow(2, 4+actionVector.at(sta_id));
			//std::cout << "MyExecuteActions-CW: " << CW[sta_id] << " da Sta "<< sta_id << std::endl;
		}
		else if (type == "continuous")
		{
			CW[sta_id] = pow(2, actionVector.at(sta_id) + 4);
			//std::cout << "MyExecuteActions-CW: " << CW[sta_id] << std::endl;
		}
		else if (type == "direct_continuous")
		{
			CW[sta_id] = actionVector.at(sta_id);
		}
		else
		{
			std::cout << "Unsupported agent type!" << endl;
			exit(0);
		}

		CW[sta_id] = min(max_cw, max(CW[sta_id], min_cw));
		outfile << current_time << "," << CW[sta_id] << endl;

		if(!dry_run){		
		
		Config::Set(std::string("/$ns3::NodeListPriv/NodeList/") + std::to_string(sta_id) + std::string("/$ns3::Node/DeviceList/") + std::to_string(sta_id) + std::string("/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw"), UintegerValue(CW[sta_id]));
				
                Config::Set(std::string("/$ns3::NodeListPriv/NodeList/") + std::to_string(sta_id) + std::string("/$ns3::Node/DeviceList/") + std::to_string(sta_id) + std::string("/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw"), UintegerValue(CW[sta_id]));
                
                //Config::Set().GetValue();
                //std::cout << "MinCw value: " << min_cw << std::endl;
		
			// a forma de concatenar a string "+sta_id+" estava errada
			//Config::Set("/$ns3::NodeListPriv/NodeList/"+sta_id+"/$ns3::Node/DeviceList/"+sta_id+"/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(CW[sta_id]));
			//Config::Set("/$ns3::NodeListPriv/NodeList/"+sta_id+"/$ns3::Node/DeviceList/"+sta_id+"/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(CW[sta_id]));
		}
	}
        return true;
}

float MyGetReward(void)
{
    static float ticks = 0.0;
    //static uint32_t last_packets = 0;
    static uint32_t last_packets[MAX_WIFI];
    //static float last_reward = 0.0;
    static float last_station_reward[MAX_WIFI];
    static float last_global_reward = 0;
    float global_reward = 0;
    ticks += envStepTime;

    for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++)
    {    
        //calculo do reward de cada estacao
        float res = g_rxPktNum[sta_id] - last_packets[sta_id];
        //std::cout << "MyGetReward-res: " << res << std::endl;
        float sta_reward = res * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024 / (5 * 150 * envStepTime) * 10; 
        //std::cout << "MyGetReward-sta_reward: " << sta_reward << "from "<< sta_id<< std::endl;

        last_packets[sta_id] = g_rxPktNum[sta_id];
        //std::cout << "MyGetReward-last_packets: " << last_packets[sta_id] << " from "<< sta_id << std::endl;

        if(verbose){
            NS_LOG_UNCOND("Station local reward: " << sta_reward);
            //std::cout << "MyGetReward Station local reward: " << sta_reward << " from "<< sta_id << std::endl;
            }

        if(sta_reward > 1.0f || sta_reward < 0.0f){
            sta_reward = last_station_reward[sta_id];
            //std::cout << "MyGetReward sta_reward: " << sta_reward << "from "<< sta_id << std::endl;
            }
        last_station_reward[sta_id] = sta_reward;
        //std::cout << "MyGetReward last_station_reward: " << last_station_reward[sta_id] << "from "<< sta_id << std::endl;            

        global_reward += sta_reward;// faz a soma total dos rewards de cada estação
        //std::cout << "MyGetReward-Soma-global_reward: " << global_reward << std::endl;
    }

    global_reward /= nWifi; // a soma total é dividida pelo numero de nodes
    //std::cout << "MyGetReward-Div-global_reward: " << global_reward << std::endl;

    if (ticks <= 2 * envStepTime)
        return 0.0; 

    if (verbose){
        NS_LOG_UNCOND("MyGetReward - Global reward:" << global_reward);
        //std::cout << "MyGetReward-verbose-global_reward: " << global_reward << std::endl;
        } 

    if(global_reward>1.0f || global_reward<0.0f){
        global_reward = last_global_reward;
        //std::cout << "MyGetReward-normaliz-global_reward: " << global_reward << std::endl;
        }
    last_global_reward = global_reward;
    //std::cout << "MyGetReward last_global_reward: " << last_global_reward << std::endl;   

    return last_global_reward;
}

/*
Collect observations
*/
Ptr<OpenGymDataContainer> MyGetObservation()
{
     recordHistory();

    //Agora uma matriz [2x300] com 2 linha e cada linha com 300 colunas:
    std::vector<uint32_t> shape = {
		history_length,
	};
	
	Ptr<OpenGymTupleContainer> data = CreateObject<OpenGymTupleContainer> ();
	
    for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++) //para percorrer/acessar as Stations
    {
		
		Ptr<OpenGymBoxContainer<float>> box = CreateObject<OpenGymBoxContainer<float>>(shape);
		
        for (uint32_t i = 0; i < history[sta_id].size(); i++) //para acessar o conteudo da Station
        {
            if (history[sta_id][i] >= -100 && history[sta_id][i] <= 100)
            {
               box->AddValue(history[sta_id][i]);               
            }
            else
               box->AddValue(0);
        }
		
		for (uint32_t i = history[sta_id].size(); i < history_length; i++)
		{
			box->AddValue(0);
		}
		
		data->Add(box);
    }

    if (verbose)
        NS_LOG_UNCOND("MyGetObservation: " << data);
        //std::cout << "MyGetObservation: " << data << std::endl;

    return data;
}

bool MyGetGameOver(void)
{
    // bool isGameOver = (ns3::Simulator::Now().GetSeconds() > simulationTime + end_delay + 1.0);
    return false;
}

void ScheduleNextStateRead(double envStepTime, Ptr<OpenGymInterface> openGymInterface)
{
    Simulator::Schedule(Seconds(envStepTime), &ScheduleNextStateRead, envStepTime, openGymInterface);
    openGymInterface->NotifyCurrentState();
}

void recordHistory()
{
    // Keep track of the observations
    // We will define them as the error rate of the last `envStepTime` seconds
    //static uint32_t last_rx = 0;            // Previously received packets
    //static uint32_t last_tx = 0;            // Previously transmitted packets
    
    /*std::vector<uint64_t> last_rx = {
        2,
    };
    std::vector<uint64_t> last_tx = {
        2,
    };*/
    static std::vector<uint64_t> last_rx(MAX_WIFI, 0);
    static std::vector<uint64_t> last_tx(MAX_WIFI, 0);
    
    static uint32_t calls = 0;              // Number of calls to this function
    calls++;
    current_time += envStepTime;
    
    for (uint32_t sta_id = 0; sta_id < nWifi ; sta_id++)
    {

         float received = g_rxPktNum[sta_id] - last_rx[sta_id];  // Received packets since the last observation
         float sent = g_txPktNum[sta_id] - last_tx[sta_id];      // Sent (...)
         float errs = sent - received;           // Errors (...)
         float ratio;                  

         ratio = errs / sent; // calcula a pcol de cada Station
         
         //std::cout << " valor da pcol da Sta " << sta_id << " = "<< ratio << std::endl;
         //history.push_front(ratio);
         history[sta_id].push_front(ratio);
         
         //std::cout << " print da history " << history[sta_id] << std::endl;

         // Remove the oldest observation if we have filled the history
         //if (history.size() > history_length)
         if (history[sta_id].size() > history_length)
         {
             //history.pop_back();
             history[sta_id].pop_back();
         }

         // Replace the last observation with the current one
         last_rx[sta_id] = g_rxPktNum[sta_id];
         last_tx[sta_id] = g_txPktNum[sta_id];

         if (calls < history_length && non_zero_start)
         {   
            // Schedule the next observation if we are not at the end of the simulation
            Simulator::Schedule(Seconds(envStepTime), &recordHistory);
         }
         else if (calls == history_length && non_zero_start)
         {
             g_rxPktNum[sta_id] = 0;
             g_txPktNum[sta_id] = 0;
             last_rx[sta_id] = 0;
             last_tx[sta_id] = 0;
         }
   }
}

/*void packetReceived(Ptr<const Packet> packet)
{
    NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
    g_rxPktNum++;
}*/

void packetReceivedWithAck(Ptr< const Packet > packet, Ptr< Ipv4 > ipv4, uint32_t interface)
{
    NS_LOG_DEBUG("Client received a packet of " << packet->GetSize() << " bytes");
    
    // Received IP Address with Ack
    Ipv4Header ipv4Header;
    packet->PeekHeader(ipv4Header);
    Ipv4Address RxsourceAddr = ipv4Header.GetSource ();
    //std::cout <<" * packetReceived Source-Addr: " << RxsourceAddr << endl;
    uint32_t RxAddr = RxsourceAddr.Get();
    int Rxoctet4 = int (RxAddr & 0xFF)-1; //Ip Address last octet
    //std::cout <<" * packetReceived RxAddr int octeto4: " << Rxoctet4 << endl;
    //std::cout << "-----------------------------" << "\n\n" << std::endl;
    
    g_rxPktNum[Rxoctet4]++;
    
    /*std::cout << ".............." << std::endl;
    std::cout << "* packetReceived g_txPktNum - Size: " <<g_rxPktNum.size() << std::endl;
    std::cout << " * packetReceived Vec_g_rxPktNum counter1: " << g_rxPktNum[0]<< std::endl;
    std::cout << " * packetReceived Vec_g_rxPktNum counter2: " << g_rxPktNum[1]<< std::endl;
    std::cout << " * packetReceived Vec_g_rxPktNum SOMA: " << g_rxPktNum[0] + g_rxPktNum[1]<< std::endl;
    std::cout << " * packetReceived Vec_g_rxPktNum TOTAL   : " << g_rxPktNum[Rxoctet4]<< "\n" <<  std::endl;*/
}

void packetSent(Ptr< const Packet > packet, Ptr< Ipv4 > ipv4, uint32_t interface)
{
    // Transmited IP Address
    Ipv4Header ipv4Header;
    packet->PeekHeader(ipv4Header);
    Ipv4Address TxsourceAddr = ipv4Header.GetSource ();
    //std::cout <<" - packetSent Source-Addr: " << TxsourceAddr << endl;
    uint32_t TxAddr = TxsourceAddr.Get();
    int Txoctet4 = int (TxAddr & 0xFF)-1; //Ip Address last octet
    //std::cout <<" - packetSent TxAddr int octeto4: " << Txoctet4 << endl;
    //std::cout << "--------------" << "\n\n" << std::endl;
    
    g_txPktNum[Txoctet4]++;
    
    /*std::cout << ".............." << std::endl;
    std::cout << " - packetSent g_txPktNum - Size: " <<g_txPktNum.size() << std::endl;
    std::cout << " - packetSent Vec_g_txPktNum counter1: " << g_txPktNum[0]<< std::endl;
    std::cout << " - packetSent Vec_g_txPktNum counter2: " << g_txPktNum[1]<< std::endl;
    std::cout << " - packetSent Vec_g_txPktNum SOMA   : " << g_txPktNum[0] + g_txPktNum[1]<< std::endl;
    std::cout << " - packetSent Vec_g_txPktNum TOTAL  : " << g_txPktNum[Txoctet4]<< "\n" << std::endl;*/
}

void set_phy(uint32_t nWifi, int guardInterval, NodeContainer &wifiStaNode, NodeContainer &wifiApNode, YansWifiPhyHelper &phy)
{
    Ptr<MatrixPropagationLossModel> lossModel = CreateObject<MatrixPropagationLossModel>();
    lossModel->SetDefaultLoss(50);

    wifiStaNode.Create(nWifi);
    wifiApNode.Create(1);

    YansWifiChannelHelper channel = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> chan = channel.Create();
    chan->SetPropagationLossModel(lossModel);
    chan->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    phy = YansWifiPhyHelper::Default();
    phy.SetChannel(chan);

    // Set guard interval
    phy.Set("GuardInterval", TimeValue(NanoSeconds(guardInterval)));
}

void set_nodes(int channelWidth, int rng, int32_t simSeed, NodeContainer wifiStaNode, NodeContainer wifiApNode, YansWifiPhyHelper phy, WifiMacHelper mac, WifiHelper wifi, NetDeviceContainer &apDevice)
{
    // Set the access point details
    Ssid ssid = Ssid("ns3-80211ax");

    mac.SetType("ns3::StaWifiMac",
                "Ssid", SsidValue(ssid),
                "ActiveProbing", BooleanValue(false),
                "BE_MaxAmpduSize", UintegerValue(0));

    NetDeviceContainer staDevice;
    staDevice = wifi.Install(phy, mac, wifiStaNode);

    mac.SetType("ns3::ApWifiMac",
                "EnableBeaconJitter", BooleanValue(false),
                "Ssid", SsidValue(ssid));

    apDevice = wifi.Install(phy, mac, wifiApNode);

    // Set channel width
    Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/ChannelWidth", UintegerValue(channelWidth));

    // mobility.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    positionAlloc->Add(Vector(0.0, 0.0, 0.0));
    positionAlloc->Add(Vector(1.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positionAlloc);

    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");

    mobility.Install(wifiApNode);
    mobility.Install(wifiStaNode);
    /* Internet stack*/
    InternetStackHelper stack;
    stack.Install(wifiApNode);
    stack.Install(wifiStaNode);

    //Random
    if(simSeed!=-1)
        RngSeedManager::SetSeed(simSeed);
    RngSeedManager::SetRun(rng);

    Ipv4AddressHelper address;
    address.SetBase("192.168.1.0", "255.255.255.0");
    Ipv4InterfaceContainer staNodeInterface;
    Ipv4InterfaceContainer apNodeInterface;

    staNodeInterface = address.Assign(staDevice);
    apNodeInterface = address.Assign(apDevice);

    for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++) //para percorrer/acessar as Stations
    {
         if(!dry_run){
			Config::Set(std::string("/$ns3::NodeListPriv/NodeList/") + std::to_string(sta_id) + std::string("/$ns3::Node/DeviceList/") + std::to_string(sta_id) + std::string("/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw"), UintegerValue(CW[sta_id]));

Config::Set(std::string("/$ns3::NodeListPriv/NodeList/") + std::to_string(sta_id) + std::string("/$ns3::Node/DeviceList/") + std::to_string(sta_id) + std::string("/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw"), UintegerValue(CW[sta_id]));
         }
         else
    {
        // para Default CW - seta um valor fixo de CW
        NS_LOG_UNCOND("Default CW");
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MinCw", UintegerValue(16));
        Config::Set("/$ns3::NodeListPriv/NodeList/*/$ns3::Node/DeviceList/*/$ns3::WifiNetDevice/Mac/$ns3::RegularWifiMac/BE_Txop/$ns3::QosTxop/MaxCw", UintegerValue(1024));
    }
    }
    
}

void set_sim(bool tracing, bool dry_run, int warmup, uint32_t openGymPort, YansWifiPhyHelper phy, NetDeviceContainer apDevice, int end_delay, Ptr<FlowMonitor> &monitor, FlowMonitorHelper &flowmon)
{
    monitor = flowmon.InstallAll();
    monitor->SetAttribute("StartTime", TimeValue(Seconds(warmup)));

    if (tracing)
    {
        phy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
        phy.EnablePcap("cw", apDevice.Get(0));
    }

    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(openGymPort);
    openGymInterface->SetGetActionSpaceCb(MakeCallback(&MyGetActionSpace));
    openGymInterface->SetGetObservationSpaceCb(MakeCallback(&MyGetObservationSpace));
    openGymInterface->SetGetGameOverCb(MakeCallback(&MyGetGameOver));
    openGymInterface->SetGetObservationCb(MakeCallback(&MyGetObservation));
    openGymInterface->SetGetRewardCb(MakeCallback(&MyGetReward));
    openGymInterface->SetGetExtraInfoCb(MakeCallback(&MyGetExtraInfo));
    openGymInterface->SetExecuteActionsCb(MakeCallback(&MyExecuteActions));

    // if (!dry_run)
    // {
    if (non_zero_start)
    {
        Simulator::Schedule(Seconds(1.0), &recordHistory);
        Simulator::Schedule(Seconds(envStepTime * history_length + 1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
    }
    else
        Simulator::Schedule(Seconds(1.0), &ScheduleNextStateRead, envStepTime, openGymInterface);
    // }

    Simulator::Stop(Seconds(simulationTime + end_delay + 1.0 + envStepTime*(history_length+1)));

    NS_LOG_UNCOND("Simulation started");
    Simulator::Run();
}

void signalHandler(int signum)
{
    cout << "Interrupt signal " << signum << " received.\n";
    exit(signum);
}

int main(int argc, char *argv[])
{
    //int nWifi = 2; //5;
    bool tracing = false;
    bool useRts = false;
    int mcs = 11;
    int channelWidth = 20;
    int guardInterval = 800;
    string offeredLoad = "150";
    int port = 1025;
    string outputCsv = "cw.csv";
    string scenario = "basic";
    dry_run = false;

    int rng = 1;
    int warmup = 1;

    uint32_t openGymPort = 5555;
    int32_t simSeed = -1;

    signal(SIGTERM, signalHandler);
    outfile << "SimulationTime,CW" << endl;

    CommandLine cmd;
    cmd.AddValue("openGymPort", "Specify port number. Default: 5555", openGymPort);
    /*for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++) //para percorrer/acessar as Stations
    {
        cmd.AddValue("CW", "Value of Contention Window", CW[sta_id]);
    }*/
    cmd.AddValue("historyLength", "Length of history window", history_length);
    cmd.AddValue("nWifi", "Number of wifi 802.11ax STA devices", nWifi);
    cmd.AddValue("verbose", "Tell echo applications to log if true", verbose);
    cmd.AddValue("tracing", "Enable pcap tracing", tracing);
    cmd.AddValue("rng", "Number of RngRun", rng);
    cmd.AddValue("simTime", "Simulation time in seconds. Default: 10s", simulationTime);
    cmd.AddValue("envStepTime", "Step time in seconds. Default: 0.1s", envStepTime);
    cmd.AddValue("agentType", "Type of agent actions: discrete, continuous", type);
    cmd.AddValue("nonZeroStart", "Start only after history buffer is filled", non_zero_start);
    cmd.AddValue("scenario", "Scenario for analysis: basic, convergence, reaction", scenario);
    cmd.AddValue("dryRun", "Execute scenario with BEB and no agent interaction", dry_run);
    cmd.AddValue("seed", "Random seed", simSeed);

    cmd.Parse(argc, argv);
    // history_length*=2;

    NS_LOG_UNCOND("Ns3Env parameters:");
    NS_LOG_UNCOND("--nWifi: " << nWifi);
    NS_LOG_UNCOND("--simulationTime: " << simulationTime);
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
    NS_LOG_UNCOND("--envStepTime: " << envStepTime);
    NS_LOG_UNCOND("--seed: " << simSeed);
    NS_LOG_UNCOND("--agentType: " << type);
    NS_LOG_UNCOND("--scenario: " << scenario);
    NS_LOG_UNCOND("--dryRun: " << dry_run);

    if (verbose)
    {
        LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
        LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);
    }

    if (useRts)
    {
        Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", StringValue("0"));
    }

    NodeContainer wifiStaNode;
    NodeContainer wifiApNode;
    YansWifiPhyHelper phy;
    set_phy(nWifi, guardInterval, wifiStaNode, wifiApNode, phy);

    WifiMacHelper mac;
    WifiHelper wifi;

    wifi.SetStandard(WIFI_PHY_STANDARD_80211ax_5GHZ);

    std::ostringstream oss;
    oss << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue(oss.str()),
                                 "ControlMode", StringValue(oss.str()));

    //802.11ac PHY
    /*
  phy.Set ("ShortGuardEnabled", BooleanValue (0));
  wifi.SetStandard (WIFI_PHY_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
  "DataMode", StringValue ("VhtMcs8"),
  "ControlMode", StringValue ("VhtMcs8"));
 */
    //802.11n PHY
    //phy.Set ("ShortGuardEnabled", BooleanValue (1));
    //wifi.SetStandard (WIFI_PHY_STANDARD_80211n_5GHZ);
    //wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
    //                              "DataMode", StringValue ("HtMcs7"),
    //                              "ControlMode", StringValue ("HtMcs7"));

    NetDeviceContainer apDevice;
    set_nodes(channelWidth, rng, simSeed, wifiStaNode, wifiApNode, phy, mac, wifi, apDevice);

    ScenarioFactory helper = ScenarioFactory(nWifi, wifiStaNode, wifiApNode, port, offeredLoad, history_length);
    wifiScenario = helper.getScenario(scenario);

    // if (!dry_run)
    // {
    if (non_zero_start)
        end_delay = envStepTime * history_length + 1.0;
    else
        end_delay = 0.0;
    // }

    wifiScenario->installScenario(simulationTime + end_delay + envStepTime, envStepTime, MakeCallback(&packetReceivedWithAck));

    // Config::ConnectWithoutContext("/NodeList/0/ApplicationList/*/$ns3::OnOffApplication/Tx", MakeCallback(&packetSent));
    //Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxBegin", MakeCallback(&packetSent));
    
    Config::ConnectWithoutContext("/NodeList/*/$ns3::Ipv4L3Protocol/Tx", MakeCallback(&packetSent));
    
    Config::ConnectWithoutContext("/NodeList/*/$ns3::Ipv4L3Protocol/Rx", MakeCallback(&packetReceivedWithAck));

    wifiScenario->PopulateARPcache();
    Ipv4GlobalRoutingHelper::PopulateRoutingTables();


    set_sim(tracing, dry_run, warmup, openGymPort, phy, apDevice, end_delay, monitor, flowmon);

    double flowThr;
    for (uint32_t sta_id = 0; sta_id < nWifi; sta_id++)
    {
        float res =  g_rxPktNum[sta_id] * (1500 - 20 - 8 - 8) * 8.0 / 1024 / 1024;
        printf("Sent mbytes: %.2f\tThroughput: %.3f", res, res/simulationTime);
    
        ofstream myfile;
        myfile.open(outputCsv, ios::app);

        /* Contents of CSV output file
        Timestamp, CW, nWifi, RngRun, SourceIP, DestinationIP, Throughput
        */
        Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon.GetClassifier());
        std::map<FlowId, FlowMonitor::FlowStats> stats = monitor->GetFlowStats();
        for (std::map<FlowId, FlowMonitor::FlowStats>::const_iterator i = stats.begin(); i != stats.end(); ++i)
        {
            auto time = std::time(nullptr); //Get timestamp
            auto tm = *std::localtime(&time);
            Ipv4FlowClassifier::FiveTuple t = classifier->FindFlow(i->first);
            flowThr = i->second.rxBytes * 8.0 / simulationTime / 1000 / 1000;
            NS_LOG_UNCOND("Flow " << i->first << " (" << t.sourceAddress << " -> " << t.destinationAddress << ")\tThroughput: " << flowThr << " Mbps\tTime: " << i->second.timeLastRxPacket.GetSeconds() - i->second.timeFirstTxPacket.GetSeconds() << " s\tRx packets " << i->second.rxPackets);
            myfile << std::put_time(&tm, "%Y-%m-%d %H:%M") << "," << CW[sta_id] << "," << nWifi << "," << RngSeedManager::GetRun() << "," << t.sourceAddress << "," << t.destinationAddress << "," << flowThr;
            myfile << std::endl;
        }
        
        myfile.close();
        
        NS_LOG_UNCOND("Packets registered by handler: " << g_rxPktNum[sta_id] << " Packets" << endl);
    }
    

    Simulator::Destroy();
    
    return 0;
}
