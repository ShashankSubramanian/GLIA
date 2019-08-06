/*************************************************************************
 *  Copyright (c) 2016-2017.
 *  All rights reserved.
 *  This file is part of the SIBIA library.
 *
 *  SIBIA is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  SIBIA is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with SIBIA. If not, see <http://www.gnu.org/licenses/>.
 ************************************************************************/

#include "EventTimings.hpp"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <ctime>
#include <vector>
#include <map>
#include <chrono>


void accumulateTimers(std::array<PetscReal, 7>& tacc, std::array<PetscReal, 7>& tloc, PetscReal selfexec) {
	tloc[5] = selfexec;
	tacc[0] += tloc[0];
	tacc[1] += tloc[1];
	tacc[2] += tloc[2];
	tacc[3] += tloc[3];
	tacc[4] += tloc[4];
	tacc[5] += tloc[5];
	tacc[6] += tloc[6];
}

void resetTimers(std::array<PetscReal, 7>& t) {
	t[0] = 0; t[1] = 0; t[2] = 0; t[3] = 0;
	t[4] = 0; t[5] = 0; t[6] = 0; t[7] = 0;
}


Event::Event(std::string eventName, Clock::duration eventDuration)
  : name(eventName),
    duration(eventDuration),
    _timers(),
    isStarted(false),
    _barrier(false)
{
  _timers[0] = 0;
  _timers[1] = 0;
  _timers[2] = 0;
  _timers[3] = 0;
  _timers[4] = 0;
  _timers[5] = 0;
  _timers[6] = 0;
  EventRegistry::put(this);
}

Event::Event(std::string eventName, bool barrier, bool autostart)
  : name(eventName),
    _barrier(barrier),
    _timers()
{
  int nprocs;
  _timers[0] = 0;
  _timers[1] = 0;
  _timers[2] = 0;
  _timers[3] = 0;
  _timers[4] = 0;
  _timers[5] = 0;
  _timers[6] = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if(nprocs <= 1) {
    _barrier = false;
  }

  if (autostart) {
    start(_barrier);
  }
}

Event::~Event()
{
  stop(_barrier);
}

void Event::start(bool barrier)
{
  if (barrier)
    MPI_Barrier(MPI_COMM_WORLD);

  isStarted = true;
  starttime = Clock::now();
}

void Event::stop(bool barrier)
{
  if (isStarted) {
    if (barrier)
      MPI_Barrier(MPI_COMM_WORLD);

    stoptime = Clock::now();
    isStarted = false;
    duration = Clock::duration(stoptime - starttime);
    EventRegistry::put(this);
  }
}

void Event::addProp(std::string property, PetscReal value)
{
  properties[property] += value;
}

void Event::addTimings(std::array<PetscReal, 7>& timings) {
  std::memcpy(_timers.data(), timings.data(), sizeof(PetscReal) * 7);
}

Event::Clock::duration Event::getDuration()
{
  return duration;
}

// -----------------------------------------------------------------------

void EventData::put(Event* event)
{
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);

  _timers[0] += event->_timers[0];
  _timers[1] += event->_timers[1];
  _timers[2] += event->_timers[2];
  _timers[3] += event->_timers[3];
  _timers[4] += event->_timers[4];
  _timers[5] += event->_timers[5];
  _timers[6] += event->_timers[6];
  _accfft_total_time += event->_timers[0] + event->_timers[4] + event->_timers[6];

  if (procid == 0) {
    count++;
    Event::Clock::duration duration = event->getDuration();
    total += duration;
    min = std::min(duration, min);
    max = std::max(duration, max);

    for (auto p : event->properties) {
      properties[p.first] += p.second;
    }
  }
}

int EventData::getAvg()
{
  return (std::chrono::duration_cast<std::chrono::milliseconds>(total) / count).count();

}

int EventData::getMax()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(max).count();
}

int EventData::getMin()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(min).count();
}

int EventData::getTotal()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(total).count();
}

int EventData::getCount()
{
  return count;
}


void EventData::compMaxProcTime()
{
  std::array<PetscReal, 7> gtimers{ {0,0,0,0,0,0,0} };
  MPI_Reduce(_timers.data(), gtimers.data(), 7, MPIType, MPI_MAX, 0, MPI_COMM_WORLD);
  std::memcpy(_timers.data(), gtimers.data(), sizeof(PetscReal) * 7);
}

void EventData::compMaxAccfftTotalTime()
{
  PetscReal g_accfft_total_time = 0;
  MPI_Reduce(&_accfft_total_time, &g_accfft_total_time, 1, MPIType, MPI_MAX, 0, MPI_COMM_WORLD);
  _accfft_total_time = g_accfft_total_time;
}

std::array<PetscReal, 7>& EventData::getTimers()
{
  return _timers;
}

PetscReal EventData::getMaxAccfftTotalTime()
{
  return _accfft_total_time;
}

int EventData::getTimePercentage(Event::Clock::duration globalDuration)
{
  return ((PetscReal) total.count() / globalDuration.count()) * 100;
}

// -----------------------------------------------------------------------

// Static members need to be initalized like that
std::map<std::string, EventData> EventRegistry::events;
Event::Clock::time_point EventRegistry::globalStart;
Event::Clock::time_point EventRegistry::globalStop;
bool EventRegistry::initialized = false;

void EventRegistry::initialize()
{
  globalStart = Event::Clock::now();
  initialized = true;
}

void EventRegistry::finalize(bool reduce)
{
  globalStop = Event::Clock::now();

  if(reduce){
    for (auto e : events) {
      e.second.compMaxProcTime();
      e.second.compMaxAccfftTotalTime();
    }
    initialized = false;
  }
}


void EventRegistry::clear()
{
  events.clear();
}

void EventRegistry::signal_handler(int signal)
{
  if (initialized) {
    finalize(false);
    print();
    print("EventTimings_SIG-nosync.log", true);
  }
  if (initialized) {
    finalize();
    print();
    print("EventTimings_SIG.log", true);
  }
}

void EventRegistry::put(Event* event)
{
  EventData data = events[event->name];
  data.put(event);
  events[event->name] = data;
}

void EventRegistry::print(std::ostream &out, bool terse)
{
  using std::endl;
  using std::setw; using std::setprecision;
  using std::left; using std::right;
  EventData::Properties allProps;
  Event::Clock::duration globalDuration = globalStop - globalStart;


  int procid, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (procid == 0) {
    std::time_t currentTime = std::time(nullptr);
    if (not terse) {
      out << "Run finished at " << std::asctime(std::localtime(&currentTime));

      out << "Global runtime       = "
          << std::chrono::duration_cast<std::chrono::milliseconds>(globalDuration).count() << "ms / "
          << std::chrono::duration_cast<std::chrono::seconds>(globalDuration).count() << "s" << std::endl
          << "Number of processors = " << nprocs << std::endl << std::endl;

      out << "Event                                Count   Total[ms]     Max[ms]     Min[ms]     Avg[ms]      T[%]       commTOT[ms]     locfftTOT[ms]   totalfftTOT[ms]   selfexecTOT[ms]   selfexecAVG[ms]" << endl;
      out << "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------" << endl;

      for (auto e : events) {
        out << setw(30) << left << e.first << right
            << setw(12) << e.second.getCount()
            << setw(12) << e.second.getTotal()
            << setw(12) << e.second.getMax()
            << setw(12) << e.second.getMin()
            << setw(12) << e.second.getAvg()
            << setw(10) << e.second.getTimePercentage(globalDuration)
            << setw(18) << e.second.getTimers()[1] * 1000
            << setw(18) << e.second.getTimers()[4] * 1000
            << setw(18) << e.second.getMaxAccfftTotalTime() * 1000
            << setw(18) << e.second.getTimers()[5] * 1000
            << setw(18) << e.second.getTimers()[5]/e.second.getCount() * 1000
            << "\n";
        for (auto p : e.second.properties) {
          allProps[p.first] += p.second;

          out << "  " << setw(12) << left << p.first
              << setw(12) << right << std::fixed << std::setprecision(5) << p.second
              << "\n";
        }
        out << "\n";
      }

      out << "Properties from all Events, accumulated" << "\n";
      out << "---------------------------------------" << "\n";
      for (auto a : allProps) {
        out << setw(14) << left << a.first << right
            << setw(12) << std::fixed << std::setprecision(5) << a.second << "\n";
      }
    }
    else // terse output
    {
      out << "# Run finished at: " << std::asctime(std::localtime(&currentTime))
          << "# Number of processors: " << nprocs << std::endl;

      auto global = std::chrono::duration_cast<std::chrono::milliseconds>(globalDuration).count();

      out << "# Eventname Count Total Max Min Avg T% comm locfftTOT totalfftTOT selfexecTOT selfexecAVG" << "\n";
      out << "\"GLOBAL\" "  << 1 << " "        // Eventname Count
          << global << " "  << global << " "   // Total Max
          <<  global << " "  << global << " "  // Min Avg
          << 100 << "\n";                      // T%
      for (auto e : events) {
        out << "\"" << e.first << "\" "
            << e.second.getCount() << " " << e.second.getTotal() << " "
            << e.second.getMax()   << " " << e.second.getMin()   << " "
            << e.second.getAvg()   << " " << e.second.getTimePercentage(globalDuration) << " "
            << std::scientific << std::setprecision(4) << e.second.getTimers()[1] * 1000 << " "
            << std::scientific << std::setprecision(4) << e.second.getTimers()[4] * 1000 << " "
            << std::scientific << std::setprecision(4) << e.second.getMaxAccfftTotalTime() * 1000 << " "
            << std::scientific << std::setprecision(4) << e.second.getTimers()[5] * 1000 << " "
            << std::scientific << std::setprecision(4) << e.second.getTimers()[5]/e.second.getCount() * 1000
            << "\n";
      }
    }
  }
  out << endl << std::flush;
}

void EventRegistry::print(bool terse)
{
  EventRegistry::print(std::cout, terse);
}

void EventRegistry::print(std::string filename, bool terse)
{
  std::ofstream outfile;
  outfile.open(filename, std::ios::out | std::ios::app);
  EventRegistry::print(outfile, terse);
  outfile.close();
}

void EventRegistry::printGlobalDuration()
{
  int procid;
  MPI_Comm_rank(MPI_COMM_WORLD, &procid);
  if (procid == 0) {
  Event::Clock::duration globalDuration = Event::Clock::now() - globalStart;

  std::cout << "Global Duration = "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
              globalDuration).count() << "ms" << std::endl;
  }
}

void Events_Init()
{
  EventRegistry::initialize();
}

void Events_Finalize()
{
  EventRegistry::finalize();
}

void Events_Clear()
{
  EventRegistry::clear();
}
